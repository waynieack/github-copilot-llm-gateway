import * as vscode from 'vscode';
import { GatewayProvider } from './provider';

/**
 * Extension activation
 */
export function activate(context: vscode.ExtensionContext): void {
  const provider = new GatewayProvider(context);

  const disposable = vscode.lm.registerLanguageModelChatProvider(
    'copilot-llm-gateway',
    provider
  );

  context.subscriptions.push(disposable);

  // Status bar entry so users can see connection state at a glance and
  // quickly open settings. Without this, failed model fetches were invisible
  // unless users happened to open the model picker.
  const statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBar.name = 'LLM Gateway';
  statusBar.command = 'github.copilot.llm-gateway.testConnection';
  statusBar.text = '$(sync~spin) LLM Gateway';
  statusBar.tooltip = 'Click to test the LLM Gateway connection';
  statusBar.show();
  context.subscriptions.push(statusBar);

  const refreshStatusBar = async (): Promise<void> => {
    try {
      const models = await provider.provideLanguageModelChatInformation(
        { silent: true },
        new vscode.CancellationTokenSource().token
      );
      if (models.length > 0) {
        statusBar.text = `$(check) LLM Gateway: ${models.length}`;
        statusBar.tooltip = `LLM Gateway: ${models.length} model(s) available.\nClick to re-test the connection.`;
      } else {
        statusBar.text = '$(warning) LLM Gateway: no models';
        statusBar.tooltip = 'No models reported by the inference server. Click to test the connection.';
      }
    } catch {
      statusBar.text = '$(error) LLM Gateway';
      statusBar.tooltip = 'LLM Gateway connection failed. Click to test the connection.';
    }
  };

  // Initial silent probe shortly after activation, once VS Code has settled.
  setTimeout(() => { void refreshStatusBar(); }, 1500);

  const testCommand = vscode.commands.registerCommand(
    'github.copilot.llm-gateway.testConnection',
    async () => {
      try {
        const models = await provider.provideLanguageModelChatInformation(
          { silent: false },
          new vscode.CancellationTokenSource().token
        );

        if (models.length > 0) {
          statusBar.text = `$(check) LLM Gateway: ${models.length}`;
          statusBar.tooltip = `LLM Gateway: ${models.length} model(s) available.\nClick to re-test the connection.`;
          vscode.window.showInformationMessage(
            `GitHub Copilot LLM Gateway: Successfully connected! Found ${models.length} model(s): ${models.map((m) => m.name).join(', ')}`
          );
        } else {
          statusBar.text = '$(warning) LLM Gateway: no models';
          vscode.window.showWarningMessage(
            'GitHub Copilot LLM Gateway: Connected but no models found.'
          );
        }
      } catch (error) {
        statusBar.text = '$(error) LLM Gateway';
        vscode.window.showErrorMessage(
          `GitHub Copilot LLM Gateway: Connection test failed. ${error instanceof Error ? error.message : String(error)}`
        );
      }
    }
  );

  context.subscriptions.push(testCommand);

  // Explicit "Refresh Models" command — previously users could only trigger
  // a re-fetch by editing settings, which was confusing when models
  // temporarily went missing (see issue #6).
  const refreshCommand = vscode.commands.registerCommand(
    'github.copilot.llm-gateway.refreshModels',
    async () => {
      provider.refreshModels();
      await refreshStatusBar();
    }
  );

  context.subscriptions.push(refreshCommand);
}

/**
 * Extension deactivation
 */
export function deactivate(): void {
  // no-op
}
