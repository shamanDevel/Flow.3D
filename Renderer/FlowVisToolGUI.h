#ifndef __TUM3D__FLOWVISTOOLGUI_H__
#define __TUM3D__FLOWVISTOOLGUI_H__


#pragma once

#include <imgui.h>
#include <FlowVisTool.h>

class FlowVisToolGUI
{
public:
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

#pragma region Dialogs
	static void LoadLinesDialog(FlowVisTool& flowVisTool);

	static void LoadBallsDialog(FlowVisTool& flowVisTool);

	static void SaveRenderingParamsDialog(FlowVisTool& flowVisTool);

	static void LoadRenderingParamsDialog(FlowVisTool& flowVisTool);

#ifdef Single
	static void SaveLinesDialog(FlowVisTool& flowVisTool);

	static void LoadSliceTexture(FlowVisTool& flowVisTool);

	static void LoadSeedTexture(FlowVisTool& flowVisTool);
#endif
#pragma endregion

	// GUI entry point.
	static void RenderGUI(FlowVisTool& flowVisTool, bool& resizeNextFrame, ImVec2& sceneWindowSize);

	static void DockSpace();

	static void MainMenu(FlowVisTool& flowVisTool);

	static void DatasetWindow(FlowVisTool& flowVisTool);

	static void TracingWindow(FlowVisTool& flowVisTool);

	static void ExtraWindow(FlowVisTool& flowVisTool);

#ifdef Single
	static void FTLEWindow(FlowVisTool& flowVisTool);
#endif

	static void HeatmapWindow(FlowVisTool& flowVisTool);

	static void TrajectoriesRenderingWindow(FlowVisTool& flowVisTool);

	static void GeneralRenderingWindow(FlowVisTool& flowVisTool);

	static void RaycastingWindow(FlowVisTool& flowVisTool);

	static void SceneWindow(FlowVisTool& flowVisTool, bool& resizeNextFrame, ImVec2& sceneWindowSize);

	static void ProfilerWindow(FlowVisTool& flowVisTool);

	static void StatusWindow(FlowVisTool& flowVisTool);
};

#endif