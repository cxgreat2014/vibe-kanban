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
    model_selector::{ModelSelectorConfig, PermissionPolicy},
    profile::ExecutorConfig,
};

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
}

#[async_trait]
impl StandardCodingAgentExecutor for Kimi {
    fn apply_overrides(&mut self, executor_config: &ExecutorConfig) {
        if let Some(model_id) = &executor_config.model_id {
            self.model = Some(model_id.clone());
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
        if let Some(model) = &self.model {
            harness = harness.with_model(model);
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
        if let Some(model) = &self.model {
            harness = harness.with_model(model);
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
            reasoning_id: None,
            permission_policy: Some(permission_policy),
        }
    }

    async fn discover_options(
        &self,
        _workdir: Option<&std::path::Path>,
        _repo_path: Option<&std::path::Path>,
    ) -> Result<futures::stream::BoxStream<'static, json_patch::Patch>, ExecutorError> {
        let options = ExecutorDiscoveredOptions {
            model_selector: ModelSelectorConfig {
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
}
