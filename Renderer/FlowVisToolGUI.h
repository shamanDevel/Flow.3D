#pragma once

#include <imgui.h>
#include <FlowVisTool.h>

class FlowVisToolGUI
{
public:
	static const float buttonWidth;

	static bool g_showRenderingOptionsWindow;
	static bool g_showTracingOptionsWindow;
	static bool g_showFTLEWindow;
	static bool g_showHeatmapWindow;
	static bool g_showExtraWindow;
	static bool g_showDatasetWindow;
	static bool g_show_demo_window;

	static int g_threadCount;
	static int g_lineIDOverride;

#pragma region Dialogs
	static void SaveLinesDialog(FlowVisTool& g_flowVisTool);

	static void LoadLinesDialog(FlowVisTool& g_flowVisTool);

	static void LoadBallsDialog(FlowVisTool& g_flowVisTool);

	static void SaveRenderingParamsDialog(FlowVisTool& g_flowVisTool);

	static void LoadRenderingParamsDialog(FlowVisTool& g_flowVisTool);

	static void LoadSliceTexture(FlowVisTool& g_flowVisTool);

	static void LoadColorTexture(FlowVisTool& g_flowVisTool);

	static void LoadSeedTexture(FlowVisTool& g_flowVisTool);
#pragma endregion

	static void DockSpace();

	static void MainMenu(FlowVisTool& g_flowVisTool);

	static void DatasetWindow(FlowVisTool& g_flowVisTool);

	static void TracingWindow(FlowVisTool& g_flowVisTool);

	static void ExtraWindow(FlowVisTool& g_flowVisTool);

	static void FTLEWindow(FlowVisTool& g_flowVisTool);

	static void HeatmapWindow(FlowVisTool& g_flowVisTool);

	static void RenderingWindow(FlowVisTool& g_flowVisTool);

	static void SceneWindow(FlowVisTool& g_flowVisTool, bool& resizeNextFrame, ImVec2& sceneWindowSize);

	static void RenderGUI(FlowVisTool& g_flowVisTool, bool& resizeNextFrame, ImVec2& sceneWindowSize);
};