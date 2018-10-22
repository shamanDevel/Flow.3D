#pragma once

#include <imgui.h>
#include <FlowVisTool.h>

class FlowVisToolGUI
{
public:
	//static const float buttonWidth;

	static bool g_showTrajectoriesRenderingSettingsWindow;
	static bool g_showGeneralRenderingSettingsWindow;
	static bool g_showTracingOptionsWindow;
	static bool g_showFTLEWindow;
	static bool g_showRaycastingWindow;
	static bool g_showHeatmapWindow;
	static bool g_showExtraWindow;
	static bool g_showDatasetWindow;
	static bool g_showProfilerWindow;
	static bool g_showStatusWindow;
	static bool g_show_demo_window;

	//static int g_threadCount;
	static int g_lineIDOverride;

#pragma region Dialogs
	//static void SaveLinesDialog(FlowVisTool& g_flowVisTool);

	static void LoadLinesDialog(FlowVisTool& g_flowVisTool);

	static void LoadBallsDialog(FlowVisTool& g_flowVisTool);

	static void SaveRenderingParamsDialog(FlowVisTool& g_flowVisTool);

	static void LoadRenderingParamsDialog(FlowVisTool& g_flowVisTool);

	//static void LoadSliceTexture(FlowVisTool& g_flowVisTool);

	//static void LoadSeedTexture(FlowVisTool& g_flowVisTool);
#pragma endregion

	static void RenderGUI(FlowVisTool& g_flowVisTool, bool& resizeNextFrame, ImVec2& sceneWindowSize);

	static void DockSpace();

	static void MainMenu(FlowVisTool& g_flowVisTool);

	static void DatasetWindow(FlowVisTool& g_flowVisTool);

	static void TracingWindow(FlowVisTool& g_flowVisTool);

	static void ExtraWindow(FlowVisTool& g_flowVisTool);

	//static void FTLEWindow(FlowVisTool& g_flowVisTool);

	static void HeatmapWindow(FlowVisTool& g_flowVisTool);

	static void TrajectoriesRenderingWindow(FlowVisTool& g_flowVisTool);

	static void GeneralRenderingWindow(FlowVisTool& g_flowVisTool);

	static void RaycastingWindow(FlowVisTool& g_flowVisTool);

	static void SceneWindow(FlowVisTool& g_flowVisTool, bool& resizeNextFrame, ImVec2& sceneWindowSize);

	static void ProfilerWindow(FlowVisTool& g_flowVisTool);

	static void StatusWindow(FlowVisTool& g_flowVisTool);
};