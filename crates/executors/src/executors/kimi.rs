use std::{path::Path, sync::Arc};

use async_trait::async_trait;
use derivative::Derivative;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use ts_rs::TS;
use workspace_utils::msg_store::MsgStore;

pub use super::acp::AcpAgentHarness;
use crate::{
    approvals::ExecutorApprovalService,
    command::{CmdOverrides, CommandBuildError, CommandBuilder, apply_overrides},
    env::ExecutionEnv,
    executor_discovery::ExecutorDiscoveredOptions,
    executors::{
        AppendPrompt, AvailabilityInfo, BaseCodingAgent, ExecutorError, SpawnedChild,
        StandardCodingAgentExecutor,
    },
    logs::utils::patch,
    model_selector::{ModelInfo, ModelSelectorConfig, PermissionPolicy, ReasoningOption},
    profile::ExecutorConfig,
};

/// Parsed model entry from ~/.kimi/config.toml
#[derive(Debug, Clone)]
struct ParsedModel {
    key: String,
    display_name: String,
    supports_thinking: bool,
}

/// Parsed Kimi CLI configuration
#[derive(Debug, Clone)]
struct ParsedConfig {
    default_model: Option<String>,
    default_thinking: bool,
    models: Vec<ParsedModel>,
}

impl Default for ParsedConfig {
    fn default() -> Self {
        Self {
            default_model: None,
            default_thinking: false,
            models: Vec::new(),
        }
    }
}

fn parse_kimi_config() -> Result<ParsedConfig, ExecutorError> {
    let config_path = dirs::home_dir()
        .ok_or_else(|| ExecutorError::UnknownExecutorType("Home directory not found".into()))?
        .join(".kimi")
        .join("config.toml");

    if !config_path.exists() {
        return Ok(ParsedConfig::default());
    }

    let content = std::fs::read_to_string(&config_path).map_err(ExecutorError::Io)?;

    let value: toml::Value = toml::from_str(&content).map_err(ExecutorError::TomlDeserialize)?;

    let default_model = value
        .get("default_model")
        .and_then(toml::Value::as_str)
        .map(String::from);

    let default_thinking = value
        .get("default_thinking")
        .and_then(toml::Value::as_bool)
        .unwrap_or(false);

    let mut models = Vec::new();
    if let Some(models_table) = value.get("models").and_then(toml::Value::as_table) {
        for (model_key, model_value) in models_table {
            if let Some(model_table) = model_value.as_table() {
                let display_name = model_table
                    .get("display_name")
                    .and_then(toml::Value::as_str)
                    .or_else(|| model_table.get("model").and_then(toml::Value::as_str))
                    .unwrap_or(model_key)
                    .to_string();

                let capabilities: Vec<&str> = model_table
                    .get("capabilities")
                    .and_then(toml::Value::as_array)
                    .map(|arr| arr.iter().filter_map(toml::Value::as_str).collect())
                    .unwrap_or_default();

                let supports_thinking = capabilities.contains(&"thinking");

                models.push(ParsedModel {
                    key: model_key.clone(),
                    display_name,
                    supports_thinking,
                });
            }
        }
    }

    Ok(ParsedConfig {
        default_model,
        default_thinking,
        models,
    })
}

/// Kimi CLI executor configuration
#[derive(Derivative, Clone, Serialize, Deserialize, TS, JsonSchema)]
#[derivative(Debug, PartialEq)]
pub struct Kimi {
    #[serde(default)]
    pub append_prompt: AppendPrompt,

    /// Model key from ~/.kimi/config.toml (e.g., "kimi-code/kimi-for-coding").
    /// If not set, the default_model from config.toml is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Enable thinking mode (appends ",thinking" to the model id sent to ACP).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,

    /// Agent type (e.g., "default", "okabe", or custom agent file)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,

    /// Skills to load
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<String>>,

    /// Custom agent file path
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_file: Option<String>,

    /// YOLO mode - auto-approve all actions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub yolo: Option<bool>,

    #[serde(flatten)]
    pub cmd: CmdOverrides,

    #[serde(skip)]
    #[ts(skip)]
    #[derivative(Debug = "ignore", PartialEq = "ignore")]
    pub approvals: Option<Arc<dyn ExecutorApprovalService>>,
}

impl Kimi {
    fn base_command(&self) -> &'static str {
        "kimi"
    }

    fn build_command_builder(&self) -> Result<CommandBuilder, CommandBuildError> {
        // Use ACP mode for programmatic interaction
        // Note: kimi acp doesn't support --model, --agent, --skill flags
        // These should be configured via ~/.kimi/config.toml instead
        let builder = CommandBuilder::new(self.base_command());

        // Use ACP mode (like Gemini)
        let builder = builder.extend_params(["acp"]);

        apply_overrides(builder, &self.cmd)
    }

    fn resolve_model_id(&self) -> Option<String> {
        self.model.as_ref().map(|model| {
            if self.thinking.unwrap_or(false) {
                format!("{model},thinking")
            } else {
                model.clone()
            }
        })
    }
}

#[async_trait]
impl StandardCodingAgentExecutor for Kimi {
    fn apply_overrides(&mut self, executor_config: &ExecutorConfig) {
        if let Some(model_id) = &executor_config.model_id {
            self.model = Some(model_id.clone());
        }
        if let Some(reasoning_id) = &executor_config.reasoning_id {
            self.thinking = Some(reasoning_id == "thinking");
        }
        if let Some(permission_policy) = executor_config.permission_policy.clone() {
            self.yolo = Some(matches!(
                permission_policy,
                crate::model_selector::PermissionPolicy::Auto
            ));
        }
    }

    fn use_approvals(&mut self, approvals: Arc<dyn ExecutorApprovalService>) {
        self.approvals = Some(approvals);
    }

    async fn spawn(
        &self,
        current_dir: &Path,
        prompt: &str,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let mut harness = AcpAgentHarness::with_session_namespace("kimi_sessions");
        if let Some(model_id) = self.resolve_model_id() {
            harness = harness.with_model(model_id);
        }
        let combined_prompt = self.append_prompt.combine_prompt(prompt);
        let kimi_command = self.build_command_builder()?.build_initial()?;
        let approvals = if self.yolo.unwrap_or(false) {
            None
        } else {
            self.approvals.clone()
        };
        harness
            .spawn_with_command(
                current_dir,
                combined_prompt,
                kimi_command,
                env,
                &self.cmd,
                approvals,
            )
            .await
    }

    async fn spawn_follow_up(
        &self,
        current_dir: &Path,
        prompt: &str,
        session_id: &str,
        _reset_to_message_id: Option<&str>,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let mut harness = AcpAgentHarness::with_session_namespace("kimi_sessions");
        if let Some(model_id) = self.resolve_model_id() {
            harness = harness.with_model(model_id);
        }
        let combined_prompt = self.append_prompt.combine_prompt(prompt);
        let kimi_command = self.build_command_builder()?.build_follow_up(&[])?;
        let approvals = if self.yolo.unwrap_or(false) {
            None
        } else {
            self.approvals.clone()
        };
        harness
            .spawn_follow_up_with_command(
                current_dir,
                combined_prompt,
                session_id,
                kimi_command,
                env,
                &self.cmd,
                approvals,
            )
            .await
    }

    fn normalize_logs(
        &self,
        msg_store: Arc<MsgStore>,
        worktree_path: &Path,
    ) -> Vec<tokio::task::JoinHandle<()>> {
        super::acp::normalize_logs(msg_store, worktree_path)
    }

    fn default_mcp_config_path(&self) -> Option<std::path::PathBuf> {
        dirs::home_dir().map(|home| home.join(".kimi").join("mcp.json"))
    }

    fn get_availability_info(&self) -> AvailabilityInfo {
        // Check for login status by looking for credentials directory
        // Kimi CLI stores credentials in ~/.kimi/credentials/ directory
        if let Some(timestamp) = dirs::home_dir()
            .and_then(|home| std::fs::metadata(home.join(".kimi").join("credentials")).ok())
            .and_then(|m| m.modified().ok())
            .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
        {
            return AvailabilityInfo::LoginDetected {
                last_auth_timestamp: timestamp,
            };
        }

        let mcp_config_found = self
            .default_mcp_config_path()
            .map(|p| p.exists())
            .unwrap_or(false);

        if mcp_config_found {
            AvailabilityInfo::InstallationFound
        } else {
            AvailabilityInfo::NotFound
        }
    }

    fn get_preset_options(&self) -> ExecutorConfig {
        let permission_policy = if self.yolo.unwrap_or(false) {
            PermissionPolicy::Auto
        } else {
            PermissionPolicy::Supervised
        };

        ExecutorConfig {
            executor: BaseCodingAgent::Kimi,
            variant: None,
            model_id: self.model.clone(),
            agent_id: self.agent.clone(),
            reasoning_id: self.thinking.and_then(|t| {
                if t {
                    Some("thinking".to_string())
                } else {
                    None
                }
            }),
            permission_policy: Some(permission_policy),
        }
    }

    async fn discover_options(
        &self,
        _workdir: Option<&std::path::Path>,
        _repo_path: Option<&std::path::Path>,
    ) -> Result<futures::stream::BoxStream<'static, json_patch::Patch>, ExecutorError> {
        let config = parse_kimi_config().unwrap_or_default();

        let models: Vec<ModelInfo> = config
            .models
            .into_iter()
            .map(|m| {
                let reasoning_options = if m.supports_thinking {
                    vec![ReasoningOption {
                        id: "thinking".to_string(),
                        label: "Thinking".to_string(),
                        is_default: config.default_thinking,
                    }]
                } else {
                    vec![]
                };

                ModelInfo {
                    id: m.key,
                    name: m.display_name,
                    provider_id: None,
                    reasoning_options,
                }
            })
            .collect();

        let options = ExecutorDiscoveredOptions {
            model_selector: ModelSelectorConfig {
                models,
                default_model: config.default_model,
                permissions: vec![PermissionPolicy::Auto, PermissionPolicy::Supervised],
                ..Default::default()
            },
            ..Default::default()
        };
        Ok(Box::pin(futures::stream::once(async move {
            patch::executor_discovered_options(options)
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kimi_availability_when_installed() {
        // This test assumes kimi is installed on the dev machine
        let kimi = Kimi {
            append_prompt: AppendPrompt::default(),
            model: None,
            thinking: None,
            agent: None,
            skills: None,
            agent_file: None,
            yolo: None,
            cmd: CmdOverrides::default(),
            approvals: None,
        };
        let info = kimi.get_availability_info();
        assert!(
            info.is_available(),
            "Kimi CLI should be detected as available when installed. Got: {:?}",
            info
        );
    }

    #[test]
    fn test_kimi_default_mcp_config_path() {
        let kimi = Kimi {
            append_prompt: AppendPrompt::default(),
            model: None,
            thinking: None,
            agent: None,
            skills: None,
            agent_file: None,
            yolo: None,
            cmd: CmdOverrides::default(),
            approvals: None,
        };
        let path = kimi.default_mcp_config_path();
        assert!(path.is_some());
        assert!(path.unwrap().to_string_lossy().contains(".kimi/mcp.json"));
    }

    #[test]
    fn test_kimi_parse_config() {
        let config = parse_kimi_config();
        // If the user has a valid config, we should get at least the default model
        if let Ok(cfg) = config {
            // The test machine has kimi installed, so we expect a config
            if !cfg.models.is_empty() {
                assert!(
                    cfg.default_model.is_some(),
                    "default_model should be present when models exist"
                );
            }
        }
    }

    #[test]
    fn test_kimi_resolve_model_id() {
        let kimi = Kimi {
            append_prompt: AppendPrompt::default(),
            model: Some("kimi-code/kimi-for-coding".to_string()),
            thinking: Some(true),
            agent: None,
            skills: None,
            agent_file: None,
            yolo: None,
            cmd: CmdOverrides::default(),
            approvals: None,
        };
        assert_eq!(
            kimi.resolve_model_id(),
            Some("kimi-code/kimi-for-coding,thinking".to_string())
        );

        let kimi_no_thinking = Kimi {
            thinking: Some(false),
            ..kimi
        };
        assert_eq!(
            kimi_no_thinking.resolve_model_id(),
            Some("kimi-code/kimi-for-coding".to_string())
        );
    }
}
