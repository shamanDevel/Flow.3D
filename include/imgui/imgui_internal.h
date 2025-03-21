// dear imgui, v1.66 WIP
// (internal structures/api)

// You may use this file to debug, understand or extend ImGui features but we don't provide any guarantee of forward compatibility!
// Set:
//   #define IMGUI_DEFINE_MATH_OPERATORS
// To implement maths operators for ImVec2 (disabled by default to not collide with using IM_VEC2_CLASS_EXTRA along with your own math types+operators)

#pragma once

#ifndef IMGUI_VERSION
#error Must include imgui.h before imgui_internal.h
#endif

#include <stdio.h>      // FILE*
#include <stdlib.h>     // NULL, malloc, free, qsort, atoi, atof
#include <math.h>       // sqrtf, fabsf, fmodf, powf, floorf, ceilf, cosf, sinf
#include <limits.h>     // INT_MIN, INT_MAX

#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4251) // class 'xxx' needs to have dll-interface to be used by clients of struct 'xxx' // when IMGUI_API is set to__declspec(dllexport)
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"        // for stb_textedit.h
#pragma clang diagnostic ignored "-Wmissing-prototypes"     // for stb_textedit.h
#pragma clang diagnostic ignored "-Wold-style-cast"
#endif

//-----------------------------------------------------------------------------
// Forward Declarations
//-----------------------------------------------------------------------------

struct ImRect;                      // An axis-aligned rectangle (2 points)
struct ImDrawDataBuilder;           // Helper to build a ImDrawData instance
struct ImDrawListSharedData;        // Data shared between all ImDrawList instances
struct ImGuiColorMod;               // Stacked color modifier, backup of modified data so we can restore it
struct ImGuiColumnData;             // Storage data for a single column
struct ImGuiColumnsSet;             // Storage data for a columns set
struct ImGuiContext;                // Main imgui context
struct ImGuiDockContext;            // Docking system context
struct ImGuiDockNode;               // Docking system node (hold a list of Windows OR two child dock nodes)
struct ImGuiDockNodeSettings;       // Storage for a dock node in .ini file (we preserve those even if the associated dock node isn't active during the session)
struct ImGuiGroupData;              // Stacked storage data for BeginGroup()/EndGroup()
struct ImGuiInputTextState;         // Internal state of the currently focused/edited text input box
struct ImGuiItemHoveredDataBackup;  // Backup and restore IsItemHovered() internal data
struct ImGuiMenuColumns;            // Simple column measurement, currently used for MenuItem() only
struct ImGuiNavMoveResult;          // Result of a directional navigation move query result
struct ImGuiNextWindowData;         // Storage for SetNexWindow** functions
struct ImGuiPopupRef;               // Storage for current popup stack
struct ImGuiSettingsHandler;        // Storage for one type registered in the .ini file
struct ImGuiStyleMod;               // Stacked style modifier, backup of modified data so we can restore it
struct ImGuiTabBar;                 // Storage for a tab bar
struct ImGuiTabItem;                // Storage for a tab item (within a tab bar)
struct ImGuiWindow;                 // Storage for one window
struct ImGuiWindowTempData;         // Temporary storage for one window (that's the data which in theory we could ditch at the end of the frame)
struct ImGuiWindowSettings;         // Storage for window settings stored in .ini file (we keep one of those even if the actual window wasn't instanced during this session)

// Use your programming IDE "Go to definition" facility on the names of the center columns to find the actual flags/enum lists.
typedef int ImGuiLayoutType;        // -> enum ImGuiLayoutType_        // Enum: Horizontal or vertical
typedef int ImGuiButtonFlags;       // -> enum ImGuiButtonFlags_       // Flags: for ButtonEx(), ButtonBehavior()
typedef int ImGuiItemFlags;         // -> enum ImGuiItemFlags_         // Flags: for PushItemFlag()
typedef int ImGuiItemStatusFlags;   // -> enum ImGuiItemStatusFlags_   // Flags: for DC.LastItemStatusFlags
typedef int ImGuiNavHighlightFlags; // -> enum ImGuiNavHighlightFlags_ // Flags: for RenderNavHighlight()
typedef int ImGuiNavDirSourceFlags; // -> enum ImGuiNavDirSourceFlags_ // Flags: for GetNavInputAmount2d()
typedef int ImGuiNavMoveFlags;      // -> enum ImGuiNavMoveFlags_      // Flags: for navigation requests
typedef int ImGuiSeparatorFlags;    // -> enum ImGuiSeparatorFlags_    // Flags: for Separator() - internal
typedef int ImGuiSliderFlags;       // -> enum ImGuiSliderFlags_       // Flags: for SliderBehavior()

//-------------------------------------------------------------------------
// STB libraries
//-------------------------------------------------------------------------

namespace ImGuiStb
{

#undef STB_TEXTEDIT_STRING
#undef STB_TEXTEDIT_CHARTYPE
#define STB_TEXTEDIT_STRING             ImGuiInputTextState
#define STB_TEXTEDIT_CHARTYPE           ImWchar
#define STB_TEXTEDIT_GETWIDTH_NEWLINE   -1.0f
#include "imstb_textedit.h"

} // namespace ImGuiStb

//-----------------------------------------------------------------------------
// Context
//-----------------------------------------------------------------------------

#ifndef GImGui
extern IMGUI_API ImGuiContext* GImGui;  // Current implicit ImGui context pointer
#endif

// Internal Drag and Drop payload types. String starting with '_' are reserved for Dear ImGui.
#define IMGUI_PAYLOAD_TYPE_WINDOW       "_IMWINDOW"     // Payload == ImGuiWindow*

//-----------------------------------------------------------------------------
// Helpers
//-----------------------------------------------------------------------------

#define IM_PI           3.14159265358979323846f
#ifdef _WIN32
#define IM_NEWLINE      "\r\n"   // Play it nice with Windows users (2018/05 news: Microsoft announced that Notepad will finally display Unix-style carriage returns!)
#else
#define IM_NEWLINE      "\n"
#endif

#define IMGUI_DEBUG_LOG(FMT,...)        printf("[%05d] " FMT, GImGui->FrameCount, __VA_ARGS__)
#define IM_STATIC_ASSERT(_COND)         typedef char static_assertion_##__line__[(_COND)?1:-1]
#define IM_F32_TO_INT8_UNBOUND(_VAL)    ((int)((_VAL) * 255.0f + ((_VAL)>=0 ? 0.5f : -0.5f)))   // Unsaturated, for display purpose
#define IM_F32_TO_INT8_SAT(_VAL)        ((int)(ImSaturate(_VAL) * 255.0f + 0.5f))               // Saturated, always output 0..255

// Enforce cdecl calling convention for functions called by the standard library, in case compilation settings changed the default to e.g. __vectorcall
#ifdef _MSC_VER
#define IMGUI_CDECL __cdecl
#else
#define IMGUI_CDECL
#endif

// Helpers: UTF-8 <> wchar
IMGUI_API int           ImTextStrToUtf8(char* buf, int buf_size, const ImWchar* in_text, const ImWchar* in_text_end);      // return output UTF-8 bytes count
IMGUI_API int           ImTextCharFromUtf8(unsigned int* out_char, const char* in_text, const char* in_text_end);          // read one character. return input UTF-8 bytes count
IMGUI_API int           ImTextStrFromUtf8(ImWchar* buf, int buf_size, const char* in_text, const char* in_text_end, const char** in_remaining = NULL);   // return input UTF-8 bytes count
IMGUI_API int           ImTextCountCharsFromUtf8(const char* in_text, const char* in_text_end);                            // return number of UTF-8 code-points (NOT bytes count)
IMGUI_API int           ImTextCountUtf8BytesFromChar(const char* in_text, const char* in_text_end);                        // return number of bytes to express one char in UTF-8
IMGUI_API int           ImTextCountUtf8BytesFromStr(const ImWchar* in_text, const ImWchar* in_text_end);                   // return number of bytes to express string in UTF-8

// Helpers: Misc
IMGUI_API ImU32         ImHash(const void* data, int data_size, ImU32 seed = 0);    // Pass data_size==0 for zero-terminated strings
IMGUI_API void*         ImFileLoadToMemory(const char* filename, const char* file_open_mode, size_t* out_file_size = NULL, int padding_bytes = 0);
IMGUI_API FILE*         ImFileOpen(const char* filename, const char* file_open_mode);
static inline bool      ImCharIsBlankA(char c)          { return c == ' ' || c == '\t'; }
static inline bool      ImCharIsBlankW(unsigned int c)  { return c == ' ' || c == '\t' || c == 0x3000; }
static inline bool      ImIsPowerOfTwo(int v)           { return v != 0 && (v & (v - 1)) == 0; }
static inline int       ImUpperPowerOfTwo(int v)        { v--; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16; v++; return v; }
#define ImQsort         qsort

// Helpers: Geometry
IMGUI_API ImVec2        ImLineClosestPoint(const ImVec2& a, const ImVec2& b, const ImVec2& p);
IMGUI_API bool          ImTriangleContainsPoint(const ImVec2& a, const ImVec2& b, const ImVec2& c, const ImVec2& p);
IMGUI_API ImVec2        ImTriangleClosestPoint(const ImVec2& a, const ImVec2& b, const ImVec2& c, const ImVec2& p);
IMGUI_API void          ImTriangleBarycentricCoords(const ImVec2& a, const ImVec2& b, const ImVec2& c, const ImVec2& p, float& out_u, float& out_v, float& out_w);
IMGUI_API ImGuiDir      ImGetDirQuadrantFromDelta(float dx, float dy);

// Helpers: String
IMGUI_API int           ImStricmp(const char* str1, const char* str2);
IMGUI_API int           ImStrnicmp(const char* str1, const char* str2, size_t count);
IMGUI_API void          ImStrncpy(char* dst, const char* src, size_t count);
IMGUI_API char*         ImStrdup(const char* str);
IMGUI_API const char*   ImStrchrRange(const char* str_begin, const char* str_end, char c);
IMGUI_API int           ImStrlenW(const ImWchar* str);
IMGUI_API const ImWchar*ImStrbolW(const ImWchar* buf_mid_line, const ImWchar* buf_begin); // Find beginning-of-line
IMGUI_API const char*   ImStristr(const char* haystack, const char* haystack_end, const char* needle, const char* needle_end);
IMGUI_API void          ImStrTrimBlanks(char* str);
IMGUI_API const char*   ImStrSkipBlank(const char* str);
IMGUI_API int           ImFormatString(char* buf, size_t buf_size, const char* fmt, ...) IM_FMTARGS(3);
IMGUI_API int           ImFormatStringV(char* buf, size_t buf_size, const char* fmt, va_list args) IM_FMTLIST(3);
IMGUI_API const char*   ImParseFormatFindStart(const char* format);
IMGUI_API const char*   ImParseFormatFindEnd(const char* format);
IMGUI_API const char*   ImParseFormatTrimDecorations(const char* format, char* buf, int buf_size);
IMGUI_API int           ImParseFormatPrecision(const char* format, int default_value);

// Helpers: ImVec2/ImVec4 operators
// We are keeping those disabled by default so they don't leak in user space, to allow user enabling implicit cast operators between ImVec2 and their own types (using IM_VEC2_CLASS_EXTRA etc.)
// We unfortunately don't have a unary- operator for ImVec2 because this would needs to be defined inside the class itself.
#ifdef IMGUI_DEFINE_MATH_OPERATORS
static inline ImVec2 operator*(const ImVec2& lhs, const float rhs)              { return ImVec2(lhs.x*rhs, lhs.y*rhs); }
static inline ImVec2 operator/(const ImVec2& lhs, const float rhs)              { return ImVec2(lhs.x/rhs, lhs.y/rhs); }
static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs)            { return ImVec2(lhs.x+rhs.x, lhs.y+rhs.y); }
static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs)            { return ImVec2(lhs.x-rhs.x, lhs.y-rhs.y); }
static inline ImVec2 operator*(const ImVec2& lhs, const ImVec2& rhs)            { return ImVec2(lhs.x*rhs.x, lhs.y*rhs.y); }
static inline ImVec2 operator/(const ImVec2& lhs, const ImVec2& rhs)            { return ImVec2(lhs.x/rhs.x, lhs.y/rhs.y); }
static inline ImVec2& operator+=(ImVec2& lhs, const ImVec2& rhs)                { lhs.x += rhs.x; lhs.y += rhs.y; return lhs; }
static inline ImVec2& operator-=(ImVec2& lhs, const ImVec2& rhs)                { lhs.x -= rhs.x; lhs.y -= rhs.y; return lhs; }
static inline ImVec2& operator*=(ImVec2& lhs, const float rhs)                  { lhs.x *= rhs; lhs.y *= rhs; return lhs; }
static inline ImVec2& operator/=(ImVec2& lhs, const float rhs)                  { lhs.x /= rhs; lhs.y /= rhs; return lhs; }
static inline ImVec4 operator+(const ImVec4& lhs, const ImVec4& rhs)            { return ImVec4(lhs.x+rhs.x, lhs.y+rhs.y, lhs.z+rhs.z, lhs.w+rhs.w); }
static inline ImVec4 operator-(const ImVec4& lhs, const ImVec4& rhs)            { return ImVec4(lhs.x-rhs.x, lhs.y-rhs.y, lhs.z-rhs.z, lhs.w-rhs.w); }
static inline ImVec4 operator*(const ImVec4& lhs, const ImVec4& rhs)            { return ImVec4(lhs.x*rhs.x, lhs.y*rhs.y, lhs.z*rhs.z, lhs.w*rhs.w); }
#endif

// Helpers: Maths
// - Wrapper for standard libs functions. (Note that imgui_demo.cpp does _not_ use them to keep the code easy to copy)
#ifndef IMGUI_DISABLE_MATH_FUNCTIONS
static inline float  ImFabs(float x)                                            { return fabsf(x); }
static inline float  ImSqrt(float x)                                            { return sqrtf(x); }
static inline float  ImPow(float x, float y)                                    { return powf(x, y); }
static inline double ImPow(double x, double y)                                  { return pow(x, y); }
static inline float  ImFmod(float x, float y)                                   { return fmodf(x, y); }
static inline double ImFmod(double x, double y)                                 { return fmod(x, y); }
static inline float  ImCos(float x)                                             { return cosf(x); }
static inline float  ImSin(float x)                                             { return sinf(x); }
static inline float  ImAcos(float x)                                            { return acosf(x); }
static inline float  ImAtan2(float y, float x)                                  { return atan2f(y, x); }
static inline double ImAtof(const char* s)                                      { return atof(s); }
static inline float  ImFloorStd(float x)                                        { return floorf(x); }   // we already uses our own ImFloor() { return (float)(int)v } internally so the standard one wrapper is named differently (it's used by stb_truetype)
static inline float  ImCeil(float x)                                            { return ceilf(x); }
#endif
// - ImMin/ImMax/ImClamp/ImLerp/ImSwap are used by widgets which support for variety of types: signed/unsigned int/long long float/double, using templates here but we could also redefine them 6 times
template<typename T> static inline T ImMin(T lhs, T rhs)                        { return lhs < rhs ? lhs : rhs; }
template<typename T> static inline T ImMax(T lhs, T rhs)                        { return lhs >= rhs ? lhs : rhs; }
template<typename T> static inline T ImClamp(T v, T mn, T mx)                   { return (v < mn) ? mn : (v > mx) ? mx : v; }
template<typename T> static inline T ImLerp(T a, T b, float t)                  { return (T)(a + (b - a) * t); }
template<typename T> static inline void ImSwap(T& a, T& b)                      { T tmp = a; a = b; b = tmp; }
// - Misc maths helpers
static inline ImVec2 ImMin(const ImVec2& lhs, const ImVec2& rhs)                { return ImVec2(lhs.x < rhs.x ? lhs.x : rhs.x, lhs.y < rhs.y ? lhs.y : rhs.y); }
static inline ImVec2 ImMax(const ImVec2& lhs, const ImVec2& rhs)                { return ImVec2(lhs.x >= rhs.x ? lhs.x : rhs.x, lhs.y >= rhs.y ? lhs.y : rhs.y); }
static inline ImVec2 ImClamp(const ImVec2& v, const ImVec2& mn, ImVec2 mx)      { return ImVec2((v.x < mn.x) ? mn.x : (v.x > mx.x) ? mx.x : v.x, (v.y < mn.y) ? mn.y : (v.y > mx.y) ? mx.y : v.y); }
static inline ImVec2 ImLerp(const ImVec2& a, const ImVec2& b, float t)          { return ImVec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t); }
static inline ImVec2 ImLerp(const ImVec2& a, const ImVec2& b, const ImVec2& t)  { return ImVec2(a.x + (b.x - a.x) * t.x, a.y + (b.y - a.y) * t.y); }
static inline ImVec4 ImLerp(const ImVec4& a, const ImVec4& b, float t)          { return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t); }
static inline float  ImSaturate(float f)                                        { return (f < 0.0f) ? 0.0f : (f > 1.0f) ? 1.0f : f; }
static inline float  ImLengthSqr(const ImVec2& lhs)                             { return lhs.x*lhs.x + lhs.y*lhs.y; }
static inline float  ImLengthSqr(const ImVec4& lhs)                             { return lhs.x*lhs.x + lhs.y*lhs.y + lhs.z*lhs.z + lhs.w*lhs.w; }
static inline float  ImInvLength(const ImVec2& lhs, float fail_value)           { float d = lhs.x*lhs.x + lhs.y*lhs.y; if (d > 0.0f) return 1.0f / ImSqrt(d); return fail_value; }
static inline float  ImFloor(float f)                                           { return (float)(int)f; }
static inline ImVec2 ImFloor(const ImVec2& v)                                   { return ImVec2((float)(int)v.x, (float)(int)v.y); }
static inline float  ImDot(const ImVec2& a, const ImVec2& b)                    { return a.x * b.x + a.y * b.y; }
static inline ImVec2 ImRotate(const ImVec2& v, float cos_a, float sin_a)        { return ImVec2(v.x * cos_a - v.y * sin_a, v.x * sin_a + v.y * cos_a); }
static inline float  ImLinearSweep(float current, float target, float speed)    { if (current < target) return ImMin(current + speed, target); if (current > target) return ImMax(current - speed, target); return current; }
static inline ImVec2 ImMul(const ImVec2& lhs, const ImVec2& rhs)                { return ImVec2(lhs.x * rhs.x, lhs.y * rhs.y); }

// Helper: ImPool<>. Basic keyed storage for contiguous instances, slow/amortized insertion, O(1) indexable, O(Log N) queries by ID over a dense/hot buffer
// Creation/Erasure invalidate all pointers, but indexes are valid as long as the object lifetime.
template<typename T>
struct IMGUI_API ImPool
{
    ImVector<T>     Data;       // Contiguous data
    ImGuiStorage    Map;        // ID->Index
    int             FreeIdx;    // Next free idx to use

    ImPool()    { FreeIdx = 0; }
    ~ImPool()   { Clear(); }
    T*          GetByKey(ImGuiID key)               { int idx = Map.GetInt(key, -1); return (idx != -1) ? &Data[idx] : NULL; }
    T*          GetByIndex(int n)                   { return &Data[n]; }
    int         GetIndex(const T* p) const          { IM_ASSERT(p >= Data.Data && p < Data.Data + Data.Size); return (int)(p - Data.Data); }
    T*          GetOrAddByKey(ImGuiID key)          { int* p_idx = Map.GetIntRef(key, -1); if (*p_idx != -1) return &Data[*p_idx]; *p_idx = FreeIdx; return Add(); }
    void        Clear()                             { for (int n = 0; n < Map.Data.Size; n++) { int idx = Map.Data[n].val_i; if (idx != -1) Data[idx].~T(); }  Map.Clear(); Data.clear(); FreeIdx = 0; }
    T*          Add()                               { int idx = FreeIdx; if (idx == Data.Size) { Data.resize(Data.Size + 1); FreeIdx++; } else { FreeIdx = *(int*)&Data[idx]; } IM_PLACEMENT_NEW(&Data[idx]) T(); return &Data[idx]; }
    void        Remove(ImGuiID key, const T* p)     { Remove(key, GetIndex(p)); }
    void        Remove(ImGuiID key, int idx)        { Data[idx].~T(); *(int*)&Data[idx] = FreeIdx; FreeIdx = idx; Map.SetInt(key, -1); }
    void        Reserve(int capacity)               { Data.reserve(capacity); Map.Data.reserve(capacity); }
};
typedef int ImPoolIdx;

//-----------------------------------------------------------------------------
// Types
//-----------------------------------------------------------------------------

enum ImGuiButtonFlags_
{
    ImGuiButtonFlags_None                   = 0,
    ImGuiButtonFlags_Repeat                 = 1 << 0,   // hold to repeat
    ImGuiButtonFlags_PressedOnClickRelease  = 1 << 1,   // return true on click + release on same item [DEFAULT if no PressedOn* flag is set]
    ImGuiButtonFlags_PressedOnClick         = 1 << 2,   // return true on click (default requires click+release)
    ImGuiButtonFlags_PressedOnRelease       = 1 << 3,   // return true on release (default requires click+release)
    ImGuiButtonFlags_PressedOnDoubleClick   = 1 << 4,   // return true on double-click (default requires click+release)
    ImGuiButtonFlags_FlattenChildren        = 1 << 5,   // allow interactions even if a child window is overlapping
    ImGuiButtonFlags_AllowItemOverlap       = 1 << 6,   // require previous frame HoveredId to either match id or be null before being usable, use along with SetItemAllowOverlap()
    ImGuiButtonFlags_DontClosePopups        = 1 << 7,   // disable automatically closing parent popup on press // [UNUSED]
    ImGuiButtonFlags_Disabled               = 1 << 8,   // disable interactions
    ImGuiButtonFlags_AlignTextBaseLine      = 1 << 9,   // vertically align button to match text baseline - ButtonEx() only // FIXME: Should be removed and handled by SmallButton(), not possible currently because of DC.CursorPosPrevLine
    ImGuiButtonFlags_NoKeyModifiers         = 1 << 10,  // disable interaction if a key modifier is held
    ImGuiButtonFlags_NoHoldingActiveID      = 1 << 11,  // don't set ActiveId while holding the mouse (ImGuiButtonFlags_PressedOnClick only)
    ImGuiButtonFlags_PressedOnDragDropHold  = 1 << 12,  // press when held into while we are drag and dropping another item (used by e.g. tree nodes, collapsing headers)
    ImGuiButtonFlags_NoNavFocus             = 1 << 13   // don't override navigation focus when activated
};

enum ImGuiSliderFlags_
{
    ImGuiSliderFlags_None                   = 0,
    ImGuiSliderFlags_Vertical               = 1 << 0
};

enum ImGuiColumnsFlags_
{
    // Default: 0
    ImGuiColumnsFlags_None                  = 0,
    ImGuiColumnsFlags_NoBorder              = 1 << 0,   // Disable column dividers
    ImGuiColumnsFlags_NoResize              = 1 << 1,   // Disable resizing columns when clicking on the dividers
    ImGuiColumnsFlags_NoPreserveWidths      = 1 << 2,   // Disable column width preservation when adjusting columns
    ImGuiColumnsFlags_NoForceWithinWindow   = 1 << 3,   // Disable forcing columns to fit within window
    ImGuiColumnsFlags_GrowParentContentsSize= 1 << 4    // (WIP) Restore pre-1.51 behavior of extending the parent window contents size but _without affecting the columns width at all_. Will eventually remove.
};

enum ImGuiSelectableFlagsPrivate_
{
    // NB: need to be in sync with last value of ImGuiSelectableFlags_
    ImGuiSelectableFlags_NoHoldingActiveID  = 1 << 10,
    ImGuiSelectableFlags_PressedOnClick     = 1 << 11,
    ImGuiSelectableFlags_PressedOnRelease   = 1 << 12,
    ImGuiSelectableFlags_DrawFillAvailWidth = 1 << 13
};

enum ImGuiSeparatorFlags_
{
    ImGuiSeparatorFlags_None                = 0,
    ImGuiSeparatorFlags_Horizontal          = 1 << 0,   // Axis default to current layout type, so generally Horizontal unless e.g. in a menu bar
    ImGuiSeparatorFlags_Vertical            = 1 << 1
};

// Transient per-window ItemFlags, reset at the beginning of the frame. For child windows: inherited from parent on first Begin().
// This is going to be exposed in imgui.h when stabilized enough.
enum ImGuiItemFlags_
{
    ImGuiItemFlags_AllowKeyboardFocus           = 1 << 0,  // true
    ImGuiItemFlags_ButtonRepeat                 = 1 << 1,  // false    // Button() will return true multiple times based on io.KeyRepeatDelay and io.KeyRepeatRate settings.
    ImGuiItemFlags_Disabled                     = 1 << 2,  // false    // [BETA] Disable interactions but doesn't affect visuals yet. See github.com/ocornut/imgui/issues/211
    ImGuiItemFlags_NoNav                        = 1 << 3,  // false
    ImGuiItemFlags_NoNavDefaultFocus            = 1 << 4,  // false
    ImGuiItemFlags_SelectableDontClosePopup     = 1 << 5,  // false    // MenuItem/Selectable() automatically closes current Popup window
    ImGuiItemFlags_Default_                     = ImGuiItemFlags_AllowKeyboardFocus
};

// Storage for LastItem data
enum ImGuiItemStatusFlags_
{
    ImGuiItemStatusFlags_None               = 0,
    ImGuiItemStatusFlags_HoveredRect        = 1 << 0,
    ImGuiItemStatusFlags_HasDisplayRect     = 1 << 1,
    ImGuiItemStatusFlags_Edited             = 1 << 2    // Value exposed by item was edited in the current frame (should match the bool return value of most widgets)
};

// FIXME: this is in development, not exposed/functional as a generic feature yet.
enum ImGuiLayoutType_
{
    ImGuiLayoutType_Vertical,
    ImGuiLayoutType_Horizontal
};

enum ImGuiAxis
{
    ImGuiAxis_None = -1,
    ImGuiAxis_X = 0,
    ImGuiAxis_Y = 1
};

enum ImGuiPlotType
{
    ImGuiPlotType_Lines,
    ImGuiPlotType_Histogram
};

enum ImGuiInputSource
{
    ImGuiInputSource_None = 0,
    ImGuiInputSource_Mouse,
    ImGuiInputSource_Nav,
    ImGuiInputSource_NavKeyboard,   // Only used occasionally for storage, not tested/handled by most code
    ImGuiInputSource_NavGamepad,    // "
    ImGuiInputSource_COUNT
};

// FIXME-NAV: Clarify/expose various repeat delay/rate
enum ImGuiInputReadMode
{
    ImGuiInputReadMode_Down,
    ImGuiInputReadMode_Pressed,
    ImGuiInputReadMode_Released,
    ImGuiInputReadMode_Repeat,
    ImGuiInputReadMode_RepeatSlow,
    ImGuiInputReadMode_RepeatFast
};

enum ImGuiNavHighlightFlags_
{
    ImGuiNavHighlightFlags_None         = 0,
    ImGuiNavHighlightFlags_TypeDefault  = 1 << 0,
    ImGuiNavHighlightFlags_TypeThin     = 1 << 1,
    ImGuiNavHighlightFlags_AlwaysDraw   = 1 << 2,
    ImGuiNavHighlightFlags_NoRounding   = 1 << 3
};

enum ImGuiNavDirSourceFlags_
{
    ImGuiNavDirSourceFlags_None         = 0,
    ImGuiNavDirSourceFlags_Keyboard     = 1 << 0,
    ImGuiNavDirSourceFlags_PadDPad      = 1 << 1,
    ImGuiNavDirSourceFlags_PadLStick    = 1 << 2
};

enum ImGuiNavMoveFlags_
{
    ImGuiNavMoveFlags_None                  = 0,
    ImGuiNavMoveFlags_LoopX                 = 1 << 0,   // On failed request, restart from opposite side
    ImGuiNavMoveFlags_LoopY                 = 1 << 1,
    ImGuiNavMoveFlags_WrapX                 = 1 << 2,   // On failed request, request from opposite side one line down (when NavDir==right) or one line up (when NavDir==left)
    ImGuiNavMoveFlags_WrapY                 = 1 << 3,   // This is not super useful for provided for completeness
    ImGuiNavMoveFlags_AllowCurrentNavId     = 1 << 4,   // Allow scoring and considering the current NavId as a move target candidate. This is used when the move source is offset (e.g. pressing PageDown actually needs to send a Up move request, if we are pressing PageDown from the bottom-most item we need to stay in place)
    ImGuiNavMoveFlags_AlsoScoreVisibleSet   = 1 << 5    // Store alternate result in NavMoveResultLocalVisibleSet that only comprise elements that are already fully visible.
};

enum ImGuiNavForward
{
    ImGuiNavForward_None,
    ImGuiNavForward_ForwardQueued,
    ImGuiNavForward_ForwardActive
};

enum ImGuiPopupPositionPolicy
{
    ImGuiPopupPositionPolicy_Default,
    ImGuiPopupPositionPolicy_ComboBox
};

// 1D vector (this odd construct is used to facilitate the transition between 1D and 2D and maintenance of some patches)
struct ImVec1
{
    float   x;
    ImVec1()         { x = 0.0f; }
    ImVec1(float _x) { x = _x; }
};

// 2D vector (half-size integer)
struct ImVec2ih
{
    short   x, y;
    ImVec2ih()                   { x = y = 0; }
    ImVec2ih(short _x, short _y) { x = _x; y = _y; }
};

// 2D axis aligned bounding-box
// NB: we can't rely on ImVec2 math operators being available here
struct IMGUI_API ImRect
{
    ImVec2      Min;    // Upper-left
    ImVec2      Max;    // Lower-right

    ImRect()                                        : Min(FLT_MAX,FLT_MAX), Max(-FLT_MAX,-FLT_MAX)  {}
    ImRect(const ImVec2& min, const ImVec2& max)    : Min(min), Max(max)                            {}
    ImRect(const ImVec4& v)                         : Min(v.x, v.y), Max(v.z, v.w)                  {}
    ImRect(float x1, float y1, float x2, float y2)  : Min(x1, y1), Max(x2, y2)                      {}

    ImVec2      GetCenter() const                   { return ImVec2((Min.x + Max.x) * 0.5f, (Min.y + Max.y) * 0.5f); }
    ImVec2      GetSize() const                     { return ImVec2(Max.x - Min.x, Max.y - Min.y); }
    float       GetWidth() const                    { return Max.x - Min.x; }
    float       GetHeight() const                   { return Max.y - Min.y; }
    ImVec2      GetTL() const                       { return Min; }                   // Top-left
    ImVec2      GetTR() const                       { return ImVec2(Max.x, Min.y); }  // Top-right
    ImVec2      GetBL() const                       { return ImVec2(Min.x, Max.y); }  // Bottom-left
    ImVec2      GetBR() const                       { return Max; }                   // Bottom-right
    bool        Contains(const ImVec2& p) const     { return p.x     >= Min.x && p.y     >= Min.y && p.x     <  Max.x && p.y     <  Max.y; }
    bool        Contains(const ImRect& r) const     { return r.Min.x >= Min.x && r.Min.y >= Min.y && r.Max.x <= Max.x && r.Max.y <= Max.y; }
    bool        Overlaps(const ImRect& r) const     { return r.Min.y <  Max.y && r.Max.y >  Min.y && r.Min.x <  Max.x && r.Max.x >  Min.x; }
    void        Add(const ImVec2& p)                { if (Min.x > p.x)     Min.x = p.x;     if (Min.y > p.y)     Min.y = p.y;     if (Max.x < p.x)     Max.x = p.x;     if (Max.y < p.y)     Max.y = p.y; }
    void        Add(const ImRect& r)                { if (Min.x > r.Min.x) Min.x = r.Min.x; if (Min.y > r.Min.y) Min.y = r.Min.y; if (Max.x < r.Max.x) Max.x = r.Max.x; if (Max.y < r.Max.y) Max.y = r.Max.y; }
    void        Expand(const float amount)          { Min.x -= amount;   Min.y -= amount;   Max.x += amount;   Max.y += amount; }
    void        Expand(const ImVec2& amount)        { Min.x -= amount.x; Min.y -= amount.y; Max.x += amount.x; Max.y += amount.y; }
    void        Translate(const ImVec2& d)          { Min.x += d.x; Min.y += d.y; Max.x += d.x; Max.y += d.y; }
    void        TranslateX(float dx)                { Min.x += dx; Max.x += dx; }
    void        TranslateY(float dy)                { Min.y += dy; Max.y += dy; }
    void        ClipWith(const ImRect& r)           { Min = ImMax(Min, r.Min); Max = ImMin(Max, r.Max); }                   // Simple version, may lead to an inverted rectangle, which is fine for Contains/Overlaps test but not for display.
    void        ClipWithFull(const ImRect& r)       { Min = ImClamp(Min, r.Min, r.Max); Max = ImClamp(Max, r.Min, r.Max); } // Full version, ensure both points are fully clipped.
    void        Floor()                             { Min.x = (float)(int)Min.x; Min.y = (float)(int)Min.y; Max.x = (float)(int)Max.x; Max.y = (float)(int)Max.y; }
    bool        IsInverted() const                  { return Min.x > Max.x || Min.y > Max.y; }
};

// Stacked color modifier, backup of modified data so we can restore it
struct ImGuiColorMod
{
    ImGuiCol    Col;
    ImVec4      BackupValue;
};

// Stacked style modifier, backup of modified data so we can restore it. Data type inferred from the variable.
struct ImGuiStyleMod
{
    ImGuiStyleVar   VarIdx;
    union           { int BackupInt[2]; float BackupFloat[2]; };
    ImGuiStyleMod(ImGuiStyleVar idx, int v)     { VarIdx = idx; BackupInt[0] = v; }
    ImGuiStyleMod(ImGuiStyleVar idx, float v)   { VarIdx = idx; BackupFloat[0] = v; }
    ImGuiStyleMod(ImGuiStyleVar idx, ImVec2 v)  { VarIdx = idx; BackupFloat[0] = v.x; BackupFloat[1] = v.y; }
};

// Stacked storage data for BeginGroup()/EndGroup()
struct ImGuiGroupData
{
    ImVec2      BackupCursorPos;
    ImVec2      BackupCursorMaxPos;
    ImVec1      BackupIndent;
    ImVec1      BackupGroupOffset;
    ImVec2      BackupCurrentLineSize;
    float       BackupCurrentLineTextBaseOffset;
    float       BackupLogLinePosY;
    ImGuiID     BackupActiveIdIsAlive;
    bool        BackupActiveIdPreviousFrameIsAlive;
    bool        AdvanceCursor;
};

// Simple column measurement, currently used for MenuItem() only.. This is very short-sighted/throw-away code and NOT a generic helper.
struct IMGUI_API ImGuiMenuColumns
{
    int         Count;
    float       Spacing;
    float       Width, NextWidth;
    float       Pos[4], NextWidths[4];

    ImGuiMenuColumns();
    void        Update(int count, float spacing, bool clear);
    float       DeclColumns(float w0, float w1, float w2);
    float       CalcExtraSpace(float avail_w);
};

// Internal state of the currently focused/edited text input box
struct IMGUI_API ImGuiInputTextState
{
    ImGuiID                 ID;                     // widget id owning the text state
    ImVector<ImWchar>       TextW;                  // edit buffer, we need to persist but can't guarantee the persistence of the user-provided buffer. so we copy into own buffer.
    ImVector<char>          InitialText;            // backup of end-user buffer at the time of focus (in UTF-8, unaltered)
    ImVector<char>          TempBuffer;             // temporary buffer for callback and other other operations. size=capacity.
    int                     CurLenA, CurLenW;       // we need to maintain our buffer length in both UTF-8 and wchar format.
    int                     BufCapacityA;           // end-user buffer capacity
    float                   ScrollX;
    ImGuiStb::STB_TexteditState StbState;
    float                   CursorAnim;
    bool                    CursorFollow;
    bool                    SelectedAllMouseLock;

    // Temporarily set when active
    ImGuiInputTextFlags     UserFlags;
    ImGuiInputTextCallback  UserCallback;
    void*                   UserCallbackData;

    ImGuiInputTextState()                           { memset(this, 0, sizeof(*this)); }
    void                CursorAnimReset()           { CursorAnim = -0.30f; }                                   // After a user-input the cursor stays on for a while without blinking
    void                CursorClamp()               { StbState.cursor = ImMin(StbState.cursor, CurLenW); StbState.select_start = ImMin(StbState.select_start, CurLenW); StbState.select_end = ImMin(StbState.select_end, CurLenW); }
    bool                HasSelection() const        { return StbState.select_start != StbState.select_end; }
    void                ClearSelection()            { StbState.select_start = StbState.select_end = StbState.cursor; }
    void                SelectAll()                 { StbState.select_start = 0; StbState.cursor = StbState.select_end = CurLenW; StbState.has_preferred_x = false; }
    void                OnKeyPressed(int key);      // Cannot be inline because we call in code in stb_textedit.h implementation
};

// Windows data saved in imgui.ini file
struct ImGuiWindowSettings
{
    char*       Name;
    ImGuiID     ID;
    ImVec2      Pos;            // NB: Settings position are stored RELATIVE to the viewport! Whereas runtime ones are absolute positions.
    ImVec2      Size;
    ImVec2      ViewportPos;
    ImGuiID     ViewportId;
    ImGuiID     DockId;         // ID of last known DockNode (even if the DockNode is invisible because it has only 1 active window), or 0 if none. 
    ImGuiID     DockFamilyId;   // ID of dock family if specified
    short       DockOrder;      // Order of the last time the window was visible within its DockNode. This is used to reorder windows that are reappearing on the same frame. Same value between windows that were active and windows that were none are possible.
    bool        Collapsed;

    ImGuiWindowSettings() { Name = NULL; ID = 0; Pos = Size = ViewportPos = ImVec2(0, 0); ViewportId = DockId = DockFamilyId = 0; DockOrder = -1; Collapsed = false; }
};

struct ImGuiSettingsHandler
{
    const char* TypeName;       // Short description stored in .ini file. Disallowed characters: '[' ']'
    ImGuiID     TypeHash;       // == ImHash(TypeName, 0, 0)
    void*       (*ReadOpenFn)(ImGuiContext* ctx, ImGuiSettingsHandler* handler, const char* name);              // Read: Called when entering into a new ini entry e.g. "[Window][Name]"
    void        (*ReadLineFn)(ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line); // Read: Called for every line of text within an ini entry
    void        (*WriteAllFn)(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* out_buf);      // Write: Output every entries into 'out_buf'
    void*       UserData;

    ImGuiSettingsHandler() { memset(this, 0, sizeof(*this)); }
};

// Storage for current popup stack
struct ImGuiPopupRef
{
    ImGuiID             PopupId;        // Set on OpenPopup()
    ImGuiWindow*        Window;         // Resolved on BeginPopup() - may stay unresolved if user never calls OpenPopup()
    ImGuiWindow*        ParentWindow;   // Set on OpenPopup()
    int                 OpenFrameCount; // Set on OpenPopup()
    ImGuiID             OpenParentId;   // Set on OpenPopup(), we need this to differenciate multiple menu sets from each others (e.g. inside menu bar vs loose menu items)
    ImVec2              OpenPopupPos;   // Set on OpenPopup(), preferred popup position (typically == OpenMousePos when using mouse)
    ImVec2              OpenMousePos;   // Set on OpenPopup(), copy of mouse position at the time of opening popup
};

struct ImGuiColumnData
{
    float               OffsetNorm;         // Column start offset, normalized 0.0 (far left) -> 1.0 (far right)
    float               OffsetNormBeforeResize;
    ImGuiColumnsFlags   Flags;              // Not exposed
    ImRect              ClipRect;

    ImGuiColumnData()   { OffsetNorm = OffsetNormBeforeResize = 0.0f; Flags = 0; }
};

struct ImGuiColumnsSet
{
    ImGuiID             ID;
    ImGuiColumnsFlags   Flags;
    bool                IsFirstFrame;
    bool                IsBeingResized;
    int                 Current;
    int                 Count;
    float               MinX, MaxX;
    float               LineMinY, LineMaxY;
    float               StartPosY;          // Copy of CursorPos
    float               StartMaxPosX;       // Copy of CursorMaxPos
    ImVector<ImGuiColumnData> Columns;

    ImGuiColumnsSet()   { Clear(); }
    void Clear()
    {
        ID = 0;
        Flags = 0;
        IsFirstFrame = false;
        IsBeingResized = false;
        Current = 0;
        Count = 1;
        MinX = MaxX = 0.0f;
        LineMinY = LineMaxY = 0.0f;
        StartPosY = 0.0f;
        StartMaxPosX = 0.0f;
        Columns.clear();
    }
};

// Data shared between all ImDrawList instances
struct IMGUI_API ImDrawListSharedData
{
    ImVec2          TexUvWhitePixel;            // UV of white pixel in the atlas
    ImFont*         Font;                       // Current/default font (optional, for simplified AddText overload)
    float           FontSize;                   // Current/default font size (optional, for simplified AddText overload)
    float           CurveTessellationTol;
    ImVec4          ClipRectFullscreen;         // Value for PushClipRectFullscreen()

    // Const data
    // FIXME: Bake rounded corners fill/borders in atlas
    ImVec2          CircleVtx12[12];

    ImDrawListSharedData();
};

struct ImDrawDataBuilder
{
    ImVector<ImDrawList*>   Layers[2];           // Global layers for: regular, tooltip

    void Clear()            { for (int n = 0; n < IM_ARRAYSIZE(Layers); n++) Layers[n].resize(0); }
    void ClearFreeMemory()  { for (int n = 0; n < IM_ARRAYSIZE(Layers); n++) Layers[n].clear(); }
    IMGUI_API void FlattenIntoSingleLayer();
};

enum ImGuiViewportFlagsPrivate_
{
    ImGuiViewportFlags_CanHostOtherWindows  = 1 << 10   // Normal viewports are associated to a single window. The main viewport can host multiple windows.
};

// ImGuiViewport Private/Internals fields (cardinal sin: we are using inheritance!)
struct ImGuiViewportP : public ImGuiViewport
{
    int                 Idx;
    int                 LastFrameActive;          // Last frame number this viewport was activated by a window
    int                 LastFrameOverlayDrawList;
    int                 LastFrontMostStampCount;  // Last stamp number from when a window hosted by this viewport was made front-most (by comparing this value between two viewport we have an implicit viewport z-order
    ImGuiID             LastNameHash;
    ImVec2              LastPos;
    bool                CreatedPlatformWindow;
    float               Alpha;                    // Window opacity (when dragging dockable windows/viewports we make them transparent)
    float               LastAlpha;
    int                 PlatformMonitor;
    ImGuiWindow*        Window;                   // Set when the viewport is owned by a window
    ImDrawList*         OverlayDrawList;          // For convenience, a draw list we can render to that's always rendered last (we use it to draw software mouse cursor when io.MouseDrawCursor is set)
    ImDrawData          DrawDataP;
    ImDrawDataBuilder   DrawDataBuilder;
    ImVec2              RendererLastSize;

    ImGuiViewportP()         { Idx = -1; LastFrameActive = LastFrameOverlayDrawList = LastFrontMostStampCount = -1; LastNameHash = 0; CreatedPlatformWindow = false; Alpha = LastAlpha = 1.0f; PlatformMonitor = INT_MIN; Window = NULL; OverlayDrawList = NULL; RendererLastSize = ImVec2(-1.0f,-1.0f); }
    ~ImGuiViewportP()        { if (OverlayDrawList) IM_DELETE(OverlayDrawList); }
    ImRect  GetRect() const  { return ImRect(Pos.x, Pos.y, Pos.x + Size.x, Pos.y + Size.y); }
    ImVec2  GetCenter() const{ return ImVec2(Pos.x + Size.x * 0.5f, Pos.y + Size.y * 0.5f); }
};

struct ImGuiNavMoveResult
{
    ImGuiID       ID;           // Best candidate
    ImGuiWindow*  Window;       // Best candidate window
    float         DistBox;      // Best candidate box distance to current NavId
    float         DistCenter;   // Best candidate center distance to current NavId
    float         DistAxial;
    ImRect        RectRel;      // Best candidate bounding box in window relative space

    ImGuiNavMoveResult() { Clear(); }
    void Clear()         { ID = 0; Window = NULL; DistBox = DistCenter = DistAxial = FLT_MAX; RectRel = ImRect(); }
};

// Storage for SetNexWindow** functions
struct ImGuiNextWindowData
{
    ImGuiCond               PosCond;
    ImGuiCond               SizeCond;
    ImGuiCond               ContentSizeCond;
    ImGuiCond               CollapsedCond;
    ImGuiCond               SizeConstraintCond;
    ImGuiCond               FocusCond;
    ImGuiCond               BgAlphaCond;
    ImGuiCond               ViewportCond;
    ImGuiCond               DockCond;
    ImVec2                  PosVal;
    ImVec2                  PosPivotVal;
    ImVec2                  SizeVal;
    ImVec2                  ContentSizeVal;
    bool                    PosUndock;
    bool                    CollapsedVal;
    ImRect                  SizeConstraintRect;
    ImGuiSizeCallback       SizeCallback;
    void*                   SizeCallbackUserData;
    float                   BgAlphaVal;
    ImGuiID                 ViewportId;
    ImGuiID                 DockId;
    ImGuiDockFamily         DockFamily;
    ImVec2                  MenuBarOffsetMinVal;                // This is not exposed publicly, so we don't clear it.

    ImGuiNextWindowData()
    {
        PosCond = SizeCond = ContentSizeCond = CollapsedCond = SizeConstraintCond = FocusCond = BgAlphaCond = ViewportCond = DockCond = 0;
        PosVal = PosPivotVal = SizeVal = ImVec2(0.0f, 0.0f);
        ContentSizeVal = ImVec2(0.0f, 0.0f);
        PosUndock = CollapsedVal = false;
        SizeConstraintRect = ImRect();
        SizeCallback = NULL;
        SizeCallbackUserData = NULL;
        BgAlphaVal = FLT_MAX;
        ViewportId = DockId = 0;
        MenuBarOffsetMinVal = ImVec2(0.0f, 0.0f);
    }

    void    Clear()
    {
        PosCond = SizeCond = ContentSizeCond = CollapsedCond = SizeConstraintCond = FocusCond = BgAlphaCond = ViewportCond = DockCond = 0;
        DockFamily = ImGuiDockFamily();
    }
};

//-----------------------------------------------------------------------------
// Docking, Tabs
//-----------------------------------------------------------------------------

struct ImGuiTabBarSortItem
{
    int         Index;
    float       Width;
};

// sizeof() 108~144
struct ImGuiDockNode
{
    ImGuiID                 ID;
    ImGuiDockNodeFlags      Flags;
    ImGuiDockNode*          ParentNode;
    ImGuiDockNode*          ChildNodes[2];          // [Split node only] Child nodes (left/right or top/bottom). Consider switching to an array.
    ImVector<ImGuiWindow*>  Windows;                // Note: unordered list! Iterate TabBar->Tabs for user-order.
    ImGuiTabBar*            TabBar;
    ImVec2                  Pos;                    // Current position
    ImVec2                  Size;                   // Current size
    ImVec2                  SizeRef;                // [Split node only] Last explicitly written-to size (overridden when using a splitter affecting the node), used to calculate Size.
    int                     SplitAxis;              // [Split node only] Split axis (X or Y)
    ImGuiDockFamily         DockFamily;

    ImGuiWindow*            HostWindow;
    ImGuiWindow*            VisibleWindow;
    ImGuiDockNode*          OnlyNodeWithWindows;    // [Root node only] Set when there is a single visible node within the hierarchy
    int                     LastFrameAlive;         // Last frame number the node was updated or kept alive explicitly with DockSpace() + ImGuiDockNodeFlags_KeepAliveOnly
    int                     LastFrameActive;        // Last frame number the node was updated.
    ImGuiID                 LastFocusedNodeID;      // [Root node only] Which of our child node (any ancestor in the hierarchy) was last focused.
    ImGuiID                 SelectedTabID;          // [Tab node only] Which of our tab is selected.
    ImGuiID                 WantCloseTabID;         // [Tab node only] Set when closing a specific tab.
    bool                    InitFromFirstWindowPosSize  :1;
    bool                    InitFromFirstWindowViewport :1;
    bool                    IsVisible           :1; // Set to false when the node is hidden (usually disabled as it has no active window)
    bool                    IsDockSpace         :1; // Root node was created by a DockSpace() call.
    bool                    IsCentralNode       :1;
    bool                    HasCloseButton      :1;
    bool                    HasCollapseButton   :1;
    bool                    WantCloseAll        :1; // Set when closing all tabs at once.
    bool                    WantLockSizeOnce    :1;
    bool                    WantMouseMove       :1; // After a node extraction we need to transition toward moving the newly created host window

    ImGuiDockNode(ImGuiID id);
    ~ImGuiDockNode();
    bool                    IsRootNode() const  { return ParentNode == NULL; }
    bool                    IsSplitNode() const { return ChildNodes[0] != NULL; }
    bool                    IsLeafNode() const  { return ChildNodes[0] == NULL; }
    bool                    IsEmpty() const     { return ChildNodes[0] == NULL && Windows.Size == 0; }
    ImRect                  Rect() const        { return ImRect(Pos.x, Pos.y, Pos.x + Size.x, Pos.y + Size.y); }
};

//-----------------------------------------------------------------------------
// Main imgui context
//-----------------------------------------------------------------------------

struct ImGuiContext
{
    bool                    Initialized;
    bool                    FrameScopeActive;                   // Set by NewFrame(), cleared by EndFrame()/Render()
    bool                    FontAtlasOwnedByContext;            // Io.Fonts-> is owned by the ImGuiContext and will be destructed along with it.
    ImGuiIO                 IO;
    ImGuiPlatformIO         PlatformIO;
    ImGuiStyle              Style;
    ImFont*                 Font;                               // (Shortcut) == FontStack.empty() ? IO.Font : FontStack.back()
    float                   FontSize;                           // (Shortcut) == FontBaseSize * g.CurrentWindow->FontWindowScale == window->FontSize(). Text height for current window.
    float                   FontBaseSize;                       // (Shortcut) == IO.FontGlobalScale * Font->Scale * Font->FontSize. Base text height.
    ImDrawListSharedData    DrawListSharedData;

    double                  Time;
    int                     FrameCount;
    int                     FrameCountEnded;
    int                     FrameCountPlatformEnded;
    int                     FrameCountRendered;
    ImVector<ImGuiWindow*>  Windows;                            // Windows, sorted in display order, back to front
    ImVector<ImGuiWindow*>  WindowsFocusOrder;                  // Windows, sorted in focus order, back to front
    ImVector<ImGuiWindow*>  WindowsSortBuffer;
    ImVector<ImGuiWindow*>  CurrentWindowStack;
    ImGuiStorage            WindowsById;
    int                     WindowsActiveCount;
    int                     WindowsFrontMostStampCount;         // Every time the front-most window changes, we stamp its viewport with an incrementing counter
    ImGuiWindow*            CurrentWindow;                      // Being drawn into
    ImGuiWindow*            HoveredWindow;                      // Will catch mouse inputs
    ImGuiWindow*            HoveredRootWindow;                  // Will catch mouse inputs (for focus/move only)
    ImGuiWindow*            HoveredWindowUnderMovingWindow;
    ImGuiID                 HoveredId;                          // Hovered widget
    bool                    HoveredIdAllowOverlap;
    ImGuiID                 HoveredIdPreviousFrame;
    float                   HoveredIdTimer;
    ImGuiID                 ActiveId;                           // Active widget
    ImGuiID                 ActiveIdPreviousFrame;
    ImGuiID                 ActiveIdIsAlive;                    // Active widget has been seen this frame (we can't use a bool as the ActiveId may change within the frame)
    float                   ActiveIdTimer;
    bool                    ActiveIdIsJustActivated;            // Set at the time of activation for one frame
    bool                    ActiveIdAllowOverlap;               // Active widget allows another widget to steal active id (generally for overlapping widgets, but not always)
    bool                    ActiveIdHasBeenEdited;              // Was the value associated to the widget Edited over the course of the Active state.
    bool                    ActiveIdPreviousFrameIsAlive;
    bool                    ActiveIdPreviousFrameHasBeenEdited;
    int                     ActiveIdAllowNavDirFlags;           // Active widget allows using directional navigation (e.g. can activate a button and move away from it)
    ImVec2                  ActiveIdClickOffset;                // Clicked offset from upper-left corner, if applicable (currently only set by ButtonBehavior)
    ImGuiWindow*            ActiveIdWindow;
    ImGuiWindow*            ActiveIdPreviousFrameWindow;
    ImGuiInputSource        ActiveIdSource;                     // Activating with mouse or nav (gamepad/keyboard)
    ImGuiID                 LastActiveId;                       // Store the last non-zero ActiveId, useful for animation.
    float                   LastActiveIdTimer;                  // Store the last non-zero ActiveId timer since the beginning of activation, useful for animation.
    ImGuiWindow*            MovingWindow;                       // Track the window we clicked on (in order to preserve focus). The actually window that is moved is generally MovingWindow->RootWindow.
    ImVector<ImGuiColorMod>   ColorModifiers;                     // Stack for PushStyleColor()/PopStyleColor()
    ImVector<ImGuiStyleMod> StyleModifiers;                     // Stack for PushStyleVar()/PopStyleVar()
    ImVector<ImFont*>       FontStack;                          // Stack for PushFont()/PopFont()
    ImVector<ImGuiPopupRef> OpenPopupStack;                     // Which popups are open (persistent)
    ImVector<ImGuiPopupRef> CurrentPopupStack;                  // Which level of BeginPopup() we are in (reset every frame)
    ImGuiNextWindowData     NextWindowData;                     // Storage for SetNextWindow** functions
    bool                    NextTreeNodeOpenVal;                // Storage for SetNextTreeNode** functions
    ImGuiCond               NextTreeNodeOpenCond;

    // Viewports
    ImVector<ImGuiViewportP*> Viewports;                        // Active viewports (always 1+, and generally 1 unless multi-viewports are enabled). Each viewports hold their copy of ImDrawData. 
    ImGuiViewportP*         CurrentViewport;                    // We track changes of viewport (happening in Begin) so we can call Platform_OnChangedViewport()
    ImGuiViewportP*         MouseViewport;
    ImGuiViewportP*         MouseLastHoveredViewport;           // Last known viewport that was hovered by mouse (even if we are not hovering any viewport any more) + honoring the _NoInputs flag.
    ImGuiID                 PlatformLastFocusedViewport;        // Record of last focused platform window/viewport, when this changes we stamp the viewport as front-most

    // Navigation data (for gamepad/keyboard)
    ImGuiWindow*            NavWindow;                          // Focused window for navigation. Could be called 'FocusWindow'
    ImGuiID                 NavId;                              // Focused item for navigation
    ImGuiID                 NavActivateId;                      // ~~ (g.ActiveId == 0) && IsNavInputPressed(ImGuiNavInput_Activate) ? NavId : 0, also set when calling ActivateItem()
    ImGuiID                 NavActivateDownId;                  // ~~ IsNavInputDown(ImGuiNavInput_Activate) ? NavId : 0
    ImGuiID                 NavActivatePressedId;               // ~~ IsNavInputPressed(ImGuiNavInput_Activate) ? NavId : 0
    ImGuiID                 NavInputId;                         // ~~ IsNavInputPressed(ImGuiNavInput_Input) ? NavId : 0
    ImGuiID                 NavJustTabbedId;                    // Just tabbed to this id.
    ImGuiID                 NavJustMovedToId;                   // Just navigated to this id (result of a successfully MoveRequest)
    ImGuiID                 NavNextActivateId;                  // Set by ActivateItem(), queued until next frame
    ImGuiInputSource        NavInputSource;                     // Keyboard or Gamepad mode?
    ImRect                  NavScoringRectScreen;               // Rectangle used for scoring, in screen space. Based of window->DC.NavRefRectRel[], modified for directional navigation scoring.
    int                     NavScoringCount;                    // Metrics for debugging
    ImGuiWindow*            NavWindowingTarget;                 // When selecting a window (holding Menu+FocusPrev/Next, or equivalent of CTRL-TAB) this window is temporarily displayed front-most.
    ImGuiWindow*            NavWindowingTargetAnim;             // Record of last valid NavWindowingTarget until DimBgRatio and NavWindowingHighlightAlpha becomes 0.0f
    ImGuiWindow*            NavWindowingList;
    float                   NavWindowingTimer;
    float                   NavWindowingHighlightAlpha;
    bool                    NavWindowingToggleLayer;
    int                     NavLayer;                           // Layer we are navigating on. For now the system is hard-coded for 0=main contents and 1=menu/title bar, may expose layers later.
    int                     NavIdTabCounter;                    // == NavWindow->DC.FocusIdxTabCounter at time of NavId processing
    bool                    NavIdIsAlive;                       // Nav widget has been seen this frame ~~ NavRefRectRel is valid
    bool                    NavMousePosDirty;                   // When set we will update mouse position if (io.ConfigFlags & ImGuiConfigFlags_NavEnableSetMousePos) if set (NB: this not enabled by default)
    bool                    NavDisableHighlight;                // When user starts using mouse, we hide gamepad/keyboard highlight (NB: but they are still available, which is why NavDisableHighlight isn't always != NavDisableMouseHover)
    bool                    NavDisableMouseHover;               // When user starts using gamepad/keyboard, we hide mouse hovering highlight until mouse is touched again.
    bool                    NavAnyRequest;                      // ~~ NavMoveRequest || NavInitRequest
    bool                    NavInitRequest;                     // Init request for appearing window to select first item
    bool                    NavInitRequestFromMove;
    ImGuiID                 NavInitResultId;
    ImRect                  NavInitResultRectRel;
    bool                    NavMoveFromClampedRefRect;          // Set by manual scrolling, if we scroll to a point where NavId isn't visible we reset navigation from visible items
    bool                    NavMoveRequest;                     // Move request for this frame
    ImGuiNavMoveFlags       NavMoveRequestFlags;
    ImGuiNavForward         NavMoveRequestForward;              // None / ForwardQueued / ForwardActive (this is used to navigate sibling parent menus from a child menu)
    ImGuiDir                NavMoveDir, NavMoveDirLast;         // Direction of the move request (left/right/up/down), direction of the previous move request
    ImGuiDir                NavMoveClipDir;
    ImGuiNavMoveResult      NavMoveResultLocal;                 // Best move request candidate within NavWindow
    ImGuiNavMoveResult      NavMoveResultLocalVisibleSet;       // Best move request candidate within NavWindow that are mostly visible (when using ImGuiNavMoveFlags_AlsoScoreVisibleSet flag)
    ImGuiNavMoveResult      NavMoveResultOther;                 // Best move request candidate within NavWindow's flattened hierarchy (when using ImGuiWindowFlags_NavFlattened flag)

    // Render
    float                   DimBgRatio;                         // 0.0..1.0 animation when fading in a dimming background (for modal window and CTRL+TAB list)
    ImGuiMouseCursor        MouseCursor;

    // Drag and Drop
    bool                    DragDropActive;
    bool                    DragDropWithinSourceOrTarget;
    ImGuiDragDropFlags      DragDropSourceFlags;
    int                     DragDropSourceFrameCount;
    int                     DragDropMouseButton;
    ImGuiPayload            DragDropPayload;
    ImRect                  DragDropTargetRect;
    ImGuiID                 DragDropTargetId;
    ImGuiDragDropFlags      DragDropAcceptFlags;
    float                   DragDropAcceptIdCurrRectSurface;    // Target item surface (we resolve overlapping targets by prioritizing the smaller surface)
    ImGuiID                 DragDropAcceptIdCurr;               // Target item id (set at the time of accepting the payload)
    ImGuiID                 DragDropAcceptIdPrev;               // Target item id from previous frame (we need to store this to allow for overlapping drag and drop targets)
    int                     DragDropAcceptFrameCount;           // Last time a target expressed a desire to accept the source
    ImVector<unsigned char> DragDropPayloadBufHeap;             // We don't expose the ImVector<> directly
    unsigned char           DragDropPayloadBufLocal[8];         // Local buffer for small payloads

    // Tab bars
    ImPool<ImGuiTabBar>     TabBars;
    ImVector<ImGuiTabBar*>  CurrentTabBar;
    ImVector<ImGuiTabBarSortItem>   TabSortByWidthBuffer;

    // Widget state
    ImGuiInputTextState     InputTextState;
    ImFont                  InputTextPasswordFont;
    ImGuiID                 ScalarAsInputTextId;                // Temporary text input when CTRL+clicking on a slider, etc.
    ImGuiColorEditFlags     ColorEditOptions;                   // Store user options for color edit widgets
    ImVec4                  ColorPickerRef;
    bool                    DragCurrentAccumDirty;
    float                   DragCurrentAccum;                   // Accumulator for dragging modification. Always high-precision, not rounded by end-user precision settings
    float                   DragSpeedDefaultRatio;              // If speed == 0.0f, uses (max-min) * DragSpeedDefaultRatio
    ImVec2                  ScrollbarClickDeltaToGrabCenter;    // Distance between mouse and center of grab box, normalized in parent space. Use storage?
    int                     TooltipOverrideCount;
    ImVector<char>          PrivateClipboard;                   // If no custom clipboard handler is defined

    // Platform support
    ImVec2                  PlatformImePos, PlatformImeLastPos; // Cursor position request & last passed to the OS Input Method Editor
    ImGuiViewport*          PlatformImePosViewport;

    // Extensions
    // FIXME: We could provide an API to register one slot in an array held in ImGuiContext?
    ImGuiDockContext*       DockContext;

    // Settings
    bool                           SettingsLoaded;
    float                          SettingsDirtyTimer;          // Save .ini Settings to memory when time reaches zero
    ImGuiTextBuffer                SettingsIniData;             // In memory .ini settings
    ImVector<ImGuiSettingsHandler> SettingsHandlers;            // List of .ini settings handlers
    ImVector<ImGuiWindowSettings>  SettingsWindows;             // ImGuiWindow .ini settings entries (parsed from the last loaded .ini file and maintained on saving)

    // Logging
    bool                    LogEnabled;
    FILE*                   LogFile;                            // If != NULL log to stdout/ file
    ImGuiTextBuffer         LogClipboard;                       // Accumulation buffer when log to clipboard. This is pointer so our GImGui static constructor doesn't call heap allocators.
    int                     LogStartDepth;
    int                     LogAutoExpandMaxDepth;

    // Misc
    float                   FramerateSecPerFrame[120];          // Calculate estimate of framerate for user over the last 2 seconds.
    int                     FramerateSecPerFrameIdx;
    float                   FramerateSecPerFrameAccum;
    int                     WantCaptureMouseNextFrame;          // Explicit capture via CaptureKeyboardFromApp()/CaptureMouseFromApp() sets those flags
    int                     WantCaptureKeyboardNextFrame;
    int                     WantTextInputNextFrame;
    char                    TempBuffer[1024*3+1];               // Temporary text buffer

    ImGuiContext(ImFontAtlas* shared_font_atlas)
    {
        Initialized = false;
        FrameScopeActive = false;
        Font = NULL;
        FontSize = FontBaseSize = 0.0f;
        FontAtlasOwnedByContext = shared_font_atlas ? false : true;
        IO.Fonts = shared_font_atlas ? shared_font_atlas : IM_NEW(ImFontAtlas)();

        Time = 0.0f;
        FrameCount = 0;
        FrameCountEnded = FrameCountPlatformEnded = FrameCountRendered = -1;
        WindowsActiveCount = 0;
        WindowsFrontMostStampCount = 0;
        CurrentWindow = NULL;
        HoveredWindow = NULL;
        HoveredRootWindow = NULL;
        HoveredId = 0;
        HoveredIdAllowOverlap = false;
        HoveredIdPreviousFrame = 0;
        HoveredIdTimer = 0.0f;
        ActiveId = 0;
        ActiveIdPreviousFrame = 0;
        ActiveIdIsAlive = 0;
        ActiveIdTimer = 0.0f;
        ActiveIdIsJustActivated = false;
        ActiveIdAllowOverlap = false;
        ActiveIdHasBeenEdited = false;
        ActiveIdPreviousFrameIsAlive = false;
        ActiveIdPreviousFrameHasBeenEdited = false;
        ActiveIdAllowNavDirFlags = 0;
        ActiveIdClickOffset = ImVec2(-1,-1);
        ActiveIdWindow = ActiveIdPreviousFrameWindow = NULL;
        ActiveIdSource = ImGuiInputSource_None;
        LastActiveId = 0;
        LastActiveIdTimer = 0.0f;
        MovingWindow = NULL;
        NextTreeNodeOpenVal = false;
        NextTreeNodeOpenCond = 0;

        CurrentViewport = NULL;
        MouseViewport = MouseLastHoveredViewport = NULL;
        PlatformLastFocusedViewport = 0;

        NavWindow = NULL;
        NavId = NavActivateId = NavActivateDownId = NavActivatePressedId = NavInputId = 0;
        NavJustTabbedId = NavJustMovedToId = NavNextActivateId = 0;
        NavInputSource = ImGuiInputSource_None;
        NavScoringRectScreen = ImRect();
        NavScoringCount = 0;
        NavWindowingTarget = NavWindowingTargetAnim = NavWindowingList = NULL;
        NavWindowingTimer = NavWindowingHighlightAlpha = 0.0f;
        NavWindowingToggleLayer = false;
        NavLayer = 0;
        NavIdTabCounter = INT_MAX;
        NavIdIsAlive = false;
        NavMousePosDirty = false;
        NavDisableHighlight = true;
        NavDisableMouseHover = false;
        NavAnyRequest = false;
        NavInitRequest = false;
        NavInitRequestFromMove = false;
        NavInitResultId = 0;
        NavMoveFromClampedRefRect = false;
        NavMoveRequest = false;
        NavMoveRequestFlags = 0;
        NavMoveRequestForward = ImGuiNavForward_None;
        NavMoveDir = NavMoveDirLast = NavMoveClipDir = ImGuiDir_None;

        DimBgRatio = 0.0f;
        MouseCursor = ImGuiMouseCursor_Arrow;

        DragDropActive = DragDropWithinSourceOrTarget = false;
        DragDropSourceFlags = 0;
        DragDropSourceFrameCount = -1;
        DragDropMouseButton = -1;
        DragDropTargetId = 0;
        DragDropAcceptFlags = 0;
        DragDropAcceptIdCurrRectSurface = 0.0f;
        DragDropAcceptIdPrev = DragDropAcceptIdCurr = 0;
        DragDropAcceptFrameCount = -1;
        memset(DragDropPayloadBufLocal, 0, sizeof(DragDropPayloadBufLocal));

        ScalarAsInputTextId = 0;
        ColorEditOptions = ImGuiColorEditFlags__OptionsDefault;
        DragCurrentAccumDirty = false;
        DragCurrentAccum = 0.0f;
        DragSpeedDefaultRatio = 1.0f / 100.0f;
        ScrollbarClickDeltaToGrabCenter = ImVec2(0.0f, 0.0f);
        TooltipOverrideCount = 0;
        PlatformImePos = PlatformImeLastPos = ImVec2(FLT_MAX, FLT_MAX);
        PlatformImePosViewport = 0;

        DockContext = NULL;

        SettingsLoaded = false;
        SettingsDirtyTimer = 0.0f;

        LogEnabled = false;
        LogFile = NULL;
        LogStartDepth = 0;
        LogAutoExpandMaxDepth = 2;

        memset(FramerateSecPerFrame, 0, sizeof(FramerateSecPerFrame));
        FramerateSecPerFrameIdx = 0;
        FramerateSecPerFrameAccum = 0.0f;
        WantCaptureMouseNextFrame = WantCaptureKeyboardNextFrame = WantTextInputNextFrame = -1;
        memset(TempBuffer, 0, sizeof(TempBuffer));
    }
};

//-----------------------------------------------------------------------------
// ImGuiWindow
//-----------------------------------------------------------------------------

// Transient per-window data, reset at the beginning of the frame. This used to be called ImGuiDrawContext, hence the DC variable name in ImGuiWindow.
// FIXME: That's theory, in practice the delimitation between ImGuiWindow and ImGuiWindowTempData is quite tenuous and could be reconsidered.
struct IMGUI_API ImGuiWindowTempData
{
    ImVec2                  CursorPos;
    ImVec2                  CursorPosPrevLine;
    ImVec2                  CursorStartPos;         // Initial position in client area with padding
    ImVec2                  CursorMaxPos;           // Used to implicitly calculate the size of our contents, always growing during the frame. Turned into window->SizeContents at the beginning of next frame
    ImVec2                  CurrentLineSize;
    float                   CurrentLineTextBaseOffset;
    ImVec2                  PrevLineSize;
    float                   PrevLineTextBaseOffset;
    float                   LogLinePosY;
    int                     TreeDepth;
    ImU32                   TreeDepthMayJumpToParentOnPop; // Store a copy of !g.NavIdIsAlive for TreeDepth 0..31
    ImGuiID                 LastItemId;
    ImGuiItemStatusFlags    LastItemStatusFlags;
    ImRect                  LastItemRect;           // Interaction rect
    ImRect                  LastItemDisplayRect;    // End-user display rect (only valid if LastItemStatusFlags & ImGuiItemStatusFlags_HasDisplayRect)
    bool                    NavHideHighlightOneFrame;
    bool                    NavHasScroll;           // Set when scrolling can be used (ScrollMax > 0.0f)
    int                     NavLayerCurrent;        // Current layer, 0..31 (we currently only use 0..1)
    int                     NavLayerCurrentMask;    // = (1 << NavLayerCurrent) used by ItemAdd prior to clipping.
    int                     NavLayerActiveMask;     // Which layer have been written to (result from previous frame)
    int                     NavLayerActiveMaskNext; // Which layer have been written to (buffer for current frame)
    bool                    MenuBarAppending;       // FIXME: Remove this
    ImVec2                  MenuBarOffset;          // MenuBarOffset.x is sort of equivalent of a per-layer CursorPos.x, saved/restored as we switch to the menu bar. The only situation when MenuBarOffset.y is > 0 if when (SafeAreaPadding.y > FramePadding.y), often used on TVs.
    ImVector<ImGuiWindow*>  ChildWindows;
    ImGuiStorage*           StateStorage;
    ImGuiLayoutType         LayoutType;
    ImGuiLayoutType         ParentLayoutType;       // Layout type of parent window at the time of Begin()

    // We store the current settings outside of the vectors to increase memory locality (reduce cache misses). The vectors are rarely modified. Also it allows us to not heap allocate for short-lived windows which are not using those settings.
    ImGuiItemFlags          ItemFlags;              // == ItemFlagsStack.back() [empty == ImGuiItemFlags_Default]
    float                   ItemWidth;              // == ItemWidthStack.back(). 0.0: default, >0.0: width in pixels, <0.0: align xx pixels to the right of window
    float                   TextWrapPos;            // == TextWrapPosStack.back() [empty == -1.0f]
    ImVector<ImGuiItemFlags>ItemFlagsStack;
    ImVector<float>         ItemWidthStack;
    ImVector<float>         TextWrapPosStack;
    ImVector<ImGuiGroupData>GroupStack;
    int                     StackSizesBackup[6];    // Store size of various stacks for asserting

    ImVec1                  Indent;                 // Indentation / start position from left of window (increased by TreePush/TreePop, etc.)
    ImVec1                  GroupOffset;
    ImVec1                  ColumnsOffset;          // Offset to the current column (if ColumnsCurrent > 0). FIXME: This and the above should be a stack to allow use cases like Tree->Column->Tree. Need revamp columns API.
    ImGuiColumnsSet*        ColumnsSet;             // Current columns set

    ImGuiWindowTempData()
    {
        CursorPos = CursorPosPrevLine = CursorStartPos = CursorMaxPos = ImVec2(0.0f, 0.0f);
        CurrentLineSize = PrevLineSize = ImVec2(0.0f, 0.0f);
        CurrentLineTextBaseOffset = PrevLineTextBaseOffset = 0.0f;
        LogLinePosY = -1.0f;
        TreeDepth = 0;
        TreeDepthMayJumpToParentOnPop = 0x00;
        LastItemId = 0;
        LastItemStatusFlags = 0;
        LastItemRect = LastItemDisplayRect = ImRect();
        NavHideHighlightOneFrame = false;
        NavHasScroll = false;
        NavLayerActiveMask = NavLayerActiveMaskNext = 0x00;
        NavLayerCurrent = 0;
        NavLayerCurrentMask = 1 << 0;
        MenuBarAppending = false;
        MenuBarOffset = ImVec2(0.0f, 0.0f);
        StateStorage = NULL;
        LayoutType = ParentLayoutType = ImGuiLayoutType_Vertical;
        ItemWidth = 0.0f;
        ItemFlags = ImGuiItemFlags_Default_;
        TextWrapPos = -1.0f;
        memset(StackSizesBackup, 0, sizeof(StackSizesBackup));

        Indent = ImVec1(0.0f);
        GroupOffset = ImVec1(0.0f);
        ColumnsOffset = ImVec1(0.0f);
        ColumnsSet = NULL;
    }
};

// Storage for one window
struct IMGUI_API ImGuiWindow
{
    char*                   Name;
    ImGuiID                 ID;                                 // == ImHash(Name)
    ImGuiWindowFlags        Flags, FlagsPreviousFrame;          // See enum ImGuiWindowFlags_
    ImGuiViewportP*         Viewport;                           // Always set in Begin(), only inactive windows may have a NULL value here
    ImGuiID                 ViewportId;                         // We backup the viewport id (since the viewport may disappear or never be created if the window is inactive)
    ImVec2                  ViewportPos;                        // We backup the viewport position (since the viewport may disappear or never be created if the window is inactive)
    int                     ViewportAllowPlatformMonitorExtend; // Reset to -1 every frame (index is guaranteed to be valid between NewFrame..EndFrame), only used in the Appearing frame of a tooltip/popup to enforce clamping to a given monitor
    ImVec2                  Pos;                                // Position (always rounded-up to nearest pixel)
    ImVec2                  Size;                               // Current size (==SizeFull or collapsed title bar size)
    ImVec2                  SizeFull;                           // Size when non collapsed
    ImVec2                  SizeFullAtLastBegin;                // Copy of SizeFull at the end of Begin. This is the reference value we'll use on the next frame to decide if we need scrollbars.
    ImVec2                  SizeContents;                       // Size of contents (== extents reach of the drawing cursor) from previous frame. Include decoration, window title, border, menu, etc.
    ImVec2                  SizeContentsExplicit;               // Size of contents explicitly set by the user via SetNextWindowContentSize()
    ImVec2                  WindowPadding;                      // Window padding at the time of begin.
    float                   WindowRounding;                     // Window rounding at the time of begin.
    float                   WindowBorderSize;                   // Window border size at the time of begin.
    ImGuiID                 MoveId;                             // == window->GetID("#MOVE")
    ImGuiID                 ChildId;                            // ID of corresponding item in parent window (for navigation to return from child window to parent window)
    ImVec2                  Scroll;
    ImVec2                  ScrollTarget;                       // target scroll position. stored as cursor position with scrolling canceled out, so the highest point is always 0.0f. (FLT_MAX for no change)
    ImVec2                  ScrollTargetCenterRatio;            // 0.0f = scroll so that target position is at top, 0.5f = scroll so that target position is centered
    ImVec2                  ScrollbarSizes;                     // Size taken by scrollbars on each axis
    bool                    ScrollbarX, ScrollbarY;
    bool                    ViewportOwned;
    bool                    ViewportTryMerge;                   // Request attempt to merge into a host viewport and destroy our owned viewport
    bool                    ViewportTrySplit;                   // Request attempt to split out of a host viewport and create our owned viewport
    bool                    Active;                             // Set to true on Begin(), unless Collapsed
    bool                    WasActive;
    bool                    WriteAccessed;                      // Set to true when any widget access the current window
    bool                    Collapsed;                          // Set when collapsing window to become only title-bar
    bool                    WantCollapseToggle;
    bool                    SkipItems;                          // Set when items can safely be all clipped (e.g. window not visible or collapsed)
    bool                    Appearing;                          // Set during the frame where the window is appearing (or re-appearing)
    bool                    Hidden;                             // Do not display (== (HiddenFramesForResize > 0) ||
    bool                    HasCloseButton;                     // Set when the window has a close button (p_open != NULL)
    int                     BeginCount;                         // Number of Begin() during the current frame (generally 0 or 1, 1+ if appending via multiple Begin/End pairs)
    int                     BeginOrderWithinParent;             // Order within immediate parent window, if we are a child window. Otherwise 0.
    int                     BeginOrderWithinContext;            // Order within entire imgui context. This is mostly used for debugging submission order related issues.
    ImGuiID                 PopupId;                            // ID in the popup stack when this window is used as a popup/menu (because we use generic Name/ID for recycling)
    int                     AutoFitFramesX, AutoFitFramesY;
    bool                    AutoFitOnlyGrows;
    int                     AutoFitChildAxises;
    ImGuiDir                AutoPosLastDirection;
    int                     HiddenFramesRegular;                // Hide the window for N frames
    int                     HiddenFramesForResize;              // Hide the window for N frames while allowing items to be submitted so we can measure their size
    ImGuiCond               SetWindowPosAllowFlags;             // store acceptable condition flags for SetNextWindowPos() use.
    ImGuiCond               SetWindowSizeAllowFlags;            // store acceptable condition flags for SetNextWindowSize() use.
    ImGuiCond               SetWindowCollapsedAllowFlags;       // store acceptable condition flags for SetNextWindowCollapsed() use.
    ImGuiCond               SetWindowDockAllowFlags;            // store acceptable condition flags for SetNextWindowDock() use.
    ImVec2                  SetWindowPosVal;                    // store window position when using a non-zero Pivot (position set needs to be processed when we know the window size)
    ImVec2                  SetWindowPosPivot;                  // store window pivot for positioning. ImVec2(0,0) when positioning from top-left corner; ImVec2(0.5f,0.5f) for centering; ImVec2(1,1) for bottom right.
    ImGuiDockFamily         DockFamily;                         // set with SetNextWindowDockFamily()

    ImGuiWindowTempData     DC;                                 // Temporary per-window data, reset at the beginning of the frame. This used to be called ImGuiDrawContext, hence the "DC" variable name.
    ImVector<ImGuiID>       IDStack;                            // ID stack. ID are hashes seeded with the value at the top of the stack
    ImRect                  ClipRect;                           // Current clipping rectangle. = DrawList->clip_rect_stack.back(). Scissoring / clipping rectangle. x1, y1, x2, y2.
    ImRect                  OuterRectClipped;                   // = WindowRect just after setup in Begin(). == window->Rect() for root window.
    ImRect                  InnerMainRect, InnerClipRect;
    ImVec2ih                HitTestHoleSize, HitTestHoleOffset;
    ImRect                  ContentsRegionRect;                 // FIXME: This is currently confusing/misleading. Maximum visible content position ~~ Pos + (SizeContentsExplicit ? SizeContentsExplicit : Size - ScrollbarSizes) - CursorStartPos, per axis
    int                     LastFrameActive;                    // Last frame number the window was Active.
    float                   ItemWidthDefault;
    ImGuiMenuColumns        MenuColumns;                        // Simplified columns storage for menu items
    ImGuiStorage            StateStorage;
    ImVector<ImGuiColumnsSet> ColumnsStorage;
    float                   FontWindowScale;                    // User scale multiplier per-window
    float                   FontDpiScale;
    int                     SettingsIdx;                        // Index into SettingsWindow[] (indices are always valid as we only grow the array from the back)

    ImDrawList*             DrawList;                           // == &DrawListInst (for backward compatibility reason with code using imgui_internal.h we keep this a pointer)
    ImDrawList              DrawListInst;
    ImGuiWindow*            ParentWindow;                       // If we are a child _or_ popup window, this is pointing to our parent. Otherwise NULL.
    ImGuiWindow*            RootWindow;                         // Point to ourself or first ancestor that is not a child window.
    ImGuiWindow*            RootWindowDockStop;                 // Point to ourself or first ancestor that is not a child window. Doesn't cross through dock nodes. We use this so IsWindowFocused() can behave consistently regardless of docking state.
    ImGuiWindow*            RootWindowForTitleBarHighlight;     // Point to ourself or first ancestor which will display TitleBgActive color when this window is active.
    ImGuiWindow*            RootWindowForNav;                   // Point to ourself or first ancestor which doesn't have the NavFlattened flag.

    ImGuiWindow*            NavLastChildNavWindow;              // When going to the menu bar, we remember the child window we came from. (This could probably be made implicit if we kept g.Windows sorted by last focused including child window.)
    ImGuiID                 NavLastIds[2];                      // Last known NavId for this window, per layer (0/1)
    ImRect                  NavRectRel[2];                      // Reference rectangle, in window relative space

    // Navigation / Focus
    // FIXME-NAV: Merge all this with the new Nav system, at least the request variables should be moved to ImGuiContext
    int                     FocusIdxAllCounter;                 // Start at -1 and increase as assigned via FocusItemRegister()
    int                     FocusIdxTabCounter;                 // (same, but only count widgets which you can Tab through)
    int                     FocusIdxAllRequestCurrent;          // Item being requested for focus
    int                     FocusIdxTabRequestCurrent;          // Tab-able item being requested for focus
    int                     FocusIdxAllRequestNext;             // Item being requested for focus, for next update (relies on layout to be stable between the frame pressing TAB and the next frame)
    int                     FocusIdxTabRequestNext;             // "

    // Docking
    ImGuiDockNode*          DockNode;                           // Which node are we docked into
    ImGuiDockNode*          DockNodeAsHost;                     // Which node are we owning (for parent windows)
    ImGuiID                 DockId;                             // Backup of last valid DockNode->Id, so single value remember their dock node id
    ImGuiItemStatusFlags    DockTabItemStatusFlags;
    ImRect                  DockTabItemRect;
    short                   DockOrder;                          // Order of the last time the window was visible within its DockNode. This is used to reorder windows that are reappearing on the same frame. Same value between windows that were active and windows that were none are possible.
    bool                    DockIsActive;                       // =~ (DockNode != NULL) && (DockNode->Windows.Size > 1)
    bool                    DockTabIsVisible;                   // Is the window visible this frame? =~ is the corresponding tab selected?
    bool                    DockTabWantClose;

public:
    ImGuiWindow(ImGuiContext* context, const char* name);
    ~ImGuiWindow();

    ImGuiID     GetID(const char* str, const char* str_end = NULL);
    ImGuiID     GetID(const void* ptr);
    ImGuiID     GetIDNoKeepAlive(const char* str, const char* str_end = NULL);
    ImGuiID     GetIDNoKeepAlive(const void* ptr);
    ImGuiID     GetIDFromRectangle(const ImRect& r_abs);

    // We don't use g.FontSize because the window may be != g.CurrentWidow.
    ImRect      Rect() const                            { return ImRect(Pos.x, Pos.y, Pos.x+Size.x, Pos.y+Size.y); }
    float       CalcFontSize() const                    { return GImGui->FontBaseSize * FontWindowScale * FontDpiScale; }
    float       TitleBarHeight() const                  { return (Flags & ImGuiWindowFlags_NoTitleBar) ? 0.0f : CalcFontSize() + GImGui->Style.FramePadding.y * 2.0f; }
    ImRect      TitleBarRect() const                    { return ImRect(Pos, ImVec2(Pos.x + SizeFull.x, Pos.y + TitleBarHeight())); }
    float       MenuBarHeight() const                   { return (Flags & ImGuiWindowFlags_MenuBar) ? DC.MenuBarOffset.y + CalcFontSize() + GImGui->Style.FramePadding.y * 2.0f : 0.0f; }
    ImRect      MenuBarRect() const                     { float y1 = Pos.y + TitleBarHeight(); return ImRect(Pos.x, y1, Pos.x + SizeFull.x, y1 + MenuBarHeight()); }
};

// Backup and restore just enough data to be able to use IsItemHovered() on item A after another B in the same window has overwritten the data.
struct ImGuiItemHoveredDataBackup
{
    ImGuiID                 LastItemId;
    ImGuiItemStatusFlags    LastItemStatusFlags;
    ImRect                  LastItemRect;
    ImRect                  LastItemDisplayRect;

    ImGuiItemHoveredDataBackup() { Backup(); }
    void Backup()           { ImGuiWindow* window = GImGui->CurrentWindow; LastItemId = window->DC.LastItemId; LastItemStatusFlags = window->DC.LastItemStatusFlags; LastItemRect = window->DC.LastItemRect; LastItemDisplayRect = window->DC.LastItemDisplayRect; }
    void Restore() const    { ImGuiWindow* window = GImGui->CurrentWindow; window->DC.LastItemId = LastItemId; window->DC.LastItemStatusFlags = LastItemStatusFlags; window->DC.LastItemRect = LastItemRect; window->DC.LastItemDisplayRect = LastItemDisplayRect; }
};

//-----------------------------------------------------------------------------
// Tab Bar, Tab Item
//-----------------------------------------------------------------------------

enum ImGuiTabBarFlagsPrivate_
{
    ImGuiTabBarFlags_DockNode                       = 1 << 20,  // Part of a dock node
    ImGuiTabBarFlags_DockNodeIsDockSpace            = 1 << 21,  // Part of an explicit dockspace node node
    ImGuiTabBarFlags_IsFocused                      = 1 << 22,
    ImGuiTabBarFlags_SaveSettings                   = 1 << 23   // FIXME: Settings are handled by the docking system, this only request the tab bar to mark settings dirty when reordering tabs
};

enum ImGuiTabItemFlagsPrivate_
{
    ImGuiTabItemFlags_DockedWindow                  = 1 << 20,  // [Docking]
    ImGuiTabItemFlags_Unsorted                      = 1 << 22,  // [Docking] Trailing tabs with the _Unsorted flag will be sorted based on the DockOrder of their Window.
    ImGuiTabItemFlags_Preview                       = 1 << 21   // [Docking] Display tab shape for docking preview (height is adjusted slightly to compensate for the yet missing tab bar)
};

// Storage for one active tab item (sizeof() 32~40 bytes)
struct ImGuiTabItem
{
    ImGuiID             ID;
    ImGuiTabItemFlags   Flags;
    ImGuiWindow*        Window;             // When TabItem is part of a DockNode's TabBar, we hold on to a window.
    int                 LastFrameVisible;
    int                 LastFrameSelected;  // This allows us to infer an ordered list of the last activated tabs with little maintenance
    float               Offset;             // Position relative to beginning of tab
    float               Width;              // Width currently displayed
    float               WidthContents;      // Width of actual contents, stored during BeginTabItem() call

    ImGuiTabItem()      { ID = Flags = 0; Window = NULL; LastFrameVisible = LastFrameSelected = -1; Offset = Width = WidthContents = 0.0f; }
};

// Storage for a tab bar (sizeof() 92~96 bytes)
struct ImGuiTabBar
{
    ImVector<ImGuiTabItem> Tabs;
    ImGuiID             ID;                     // Zero for tab-bars used by docking
    ImGuiID             SelectedTabId;          // Selected tab
    ImGuiID             NextSelectedTabId;
    ImGuiID             VisibleTabId;           // Can occasionally be != SelectedTabId (e.g. when previewing contents for CTRL+TAB preview)
    ImGuiID             WantFocusTabId;         // Request focus for the window associated to this tab. Used and only honored by DockNode (meaningless for standalone tab bars)
    int                 CurrFrameVisible;
    int                 PrevFrameVisible;
    ImRect              BarRect;
    float               ContentsHeight;
    float               OffsetMax;              // Distance from BarRect.Min.x, locked during layout
    float               OffsetNextTab;          // Distance from BarRect.Min.x, incremented with each BeginTabItem() call, not used if ImGuiTabBarFlags_Reorderable if set.
    float               ScrollingAnim;
    float               ScrollingTarget;
    ImGuiTabBarFlags    Flags;
    ImGuiID             ReorderRequestTabId;
    int                 ReorderRequestDir;
    bool                WantLayout;
    bool                VisibleTabWasSubmitted;
    short               LastTabItemIdx;         // For BeginTabItem()/EndTabItem()

    ImGuiTabBar();
    int                 GetTabOrder(const ImGuiTabItem* tab) const { return Tabs.index_from_pointer(tab); }
};

//-----------------------------------------------------------------------------
// Internal API
// No guarantee of forward compatibility here.
//-----------------------------------------------------------------------------

namespace ImGui
{
    // We should always have a CurrentWindow in the stack (there is an implicit "Debug" window)
    // If this ever crash because g.CurrentWindow is NULL it means that either
    // - ImGui::NewFrame() has never been called, which is illegal.
    // - You are calling ImGui functions after ImGui::EndFrame()/ImGui::Render() and before the next ImGui::NewFrame(), which is also illegal.
    inline    ImGuiWindow*  GetCurrentWindowRead()      { ImGuiContext& g = *GImGui; return g.CurrentWindow; }
    inline    ImGuiWindow*  GetCurrentWindow()          { ImGuiContext& g = *GImGui; g.CurrentWindow->WriteAccessed = true; return g.CurrentWindow; }
    IMGUI_API ImGuiWindow*  FindWindowByID(ImGuiID id);
    IMGUI_API ImGuiWindow*  FindWindowByName(const char* name);
    IMGUI_API void          FocusWindow(ImGuiWindow* window);
    IMGUI_API void          FocusPreviousWindowIgnoringOne(ImGuiWindow* ignore_window);
    IMGUI_API void          BringWindowToFocusFront(ImGuiWindow* window);
    IMGUI_API void          BringWindowToDisplayFront(ImGuiWindow* window);
    IMGUI_API void          BringWindowToDisplayBack(ImGuiWindow* window);
    IMGUI_API void          UpdateWindowParentAndRootLinks(ImGuiWindow* window, ImGuiWindowFlags flags, ImGuiWindow* parent_window);
    IMGUI_API ImVec2        CalcWindowExpectedSize(ImGuiWindow* window);
    IMGUI_API bool          IsWindowChildOf(ImGuiWindow* window, ImGuiWindow* potential_parent);
    IMGUI_API bool          IsWindowNavFocusable(ImGuiWindow* window);
    IMGUI_API void          SetWindowScrollX(ImGuiWindow* window, float new_scroll_x);
    IMGUI_API void          SetWindowScrollY(ImGuiWindow* window, float new_scroll_y);
    IMGUI_API ImRect        GetWindowAllowedExtentRect(ImGuiWindow* window);

    IMGUI_API void          SetCurrentFont(ImFont* font);
    inline ImFont*          GetDefaultFont() { ImGuiContext& g = *GImGui; return g.IO.FontDefault ? g.IO.FontDefault : g.IO.Fonts->Fonts[0]; }
    IMGUI_API ImDrawList*   GetOverlayDrawList(ImGuiViewportP* viewport);
    inline ImDrawList*      GetOverlayDrawList(ImGuiWindow* window) { return GetOverlayDrawList(window->Viewport); }

    // Init
    IMGUI_API void          Initialize(ImGuiContext* context);
    IMGUI_API void          Shutdown(ImGuiContext* context);    // Since 1.60 this is a _private_ function. You can call DestroyContext() to destroy the context created by CreateContext().

    // NewFrame
    IMGUI_API void          UpdateHoveredWindowAndCaptureFlags();
    IMGUI_API void          StartMouseMovingWindow(ImGuiWindow* window);
    IMGUI_API void          UpdateMouseMovingWindow();

    // Viewports
    IMGUI_API ImGuiViewportP*       FindViewportByID(ImGuiID id);
    IMGUI_API void                  ScaleWindowsInViewport(ImGuiViewportP* viewport, float scale);
    IMGUI_API void                  DestroyPlatformWindow(ImGuiViewportP* viewport);
    IMGUI_API void                  ShowViewportThumbnails();

    // Settings
    IMGUI_API void                  MarkIniSettingsDirty();
    IMGUI_API void                  MarkIniSettingsDirty(ImGuiWindow* window);
    IMGUI_API ImGuiWindowSettings*  CreateNewWindowSettings(const char* name);
    IMGUI_API ImGuiWindowSettings*  FindWindowSettings(ImGuiID id);
    IMGUI_API ImGuiWindowSettings*  FindOrCreateWindowSettings(const char* name);
    IMGUI_API ImGuiSettingsHandler* FindSettingsHandler(const char* type_name);

    // Basic Accessors
    inline ImGuiID          GetItemID()     { ImGuiContext& g = *GImGui; return g.CurrentWindow->DC.LastItemId; }
    inline ImGuiID          GetActiveID()   { ImGuiContext& g = *GImGui; return g.ActiveId; }
    inline ImGuiID          GetFocusID()    { ImGuiContext& g = *GImGui; return g.NavId; }
    IMGUI_API void          SetActiveID(ImGuiID id, ImGuiWindow* window);
    IMGUI_API void          SetFocusID(ImGuiID id, ImGuiWindow* window);
    IMGUI_API void          ClearActiveID();
    IMGUI_API ImGuiID       GetHoveredID();
    IMGUI_API void          SetHoveredID(ImGuiID id);
    IMGUI_API void          KeepAliveID(ImGuiID id);
    IMGUI_API void          MarkItemEdited(ImGuiID id);

    // Basic Helpers for widget code
    IMGUI_API void          ItemSize(const ImVec2& size, float text_offset_y = 0.0f);
    IMGUI_API void          ItemSize(const ImRect& bb, float text_offset_y = 0.0f);
    IMGUI_API bool          ItemAdd(const ImRect& bb, ImGuiID id, const ImRect* nav_bb = NULL);
    IMGUI_API bool          ItemHoverable(const ImRect& bb, ImGuiID id);
    IMGUI_API bool          IsClippedEx(const ImRect& bb, ImGuiID id, bool clip_even_when_logged);
    IMGUI_API bool          FocusableItemRegister(ImGuiWindow* window, ImGuiID id, bool tab_stop = true);      // Return true if focus is requested
    IMGUI_API void          FocusableItemUnregister(ImGuiWindow* window);
    IMGUI_API ImVec2        CalcItemSize(ImVec2 size, float default_x, float default_y);
    IMGUI_API float         CalcWrapWidthForPos(const ImVec2& pos, float wrap_pos_x);
    IMGUI_API void          PushMultiItemsWidths(int components, float width_full = 0.0f);
    IMGUI_API void          PushItemFlag(ImGuiItemFlags option, bool enabled);
    IMGUI_API void          PopItemFlag();

    // Popups, Modals, Tooltips
    IMGUI_API void          OpenPopupEx(ImGuiID id);
    IMGUI_API void          ClosePopup(ImGuiID id);
    IMGUI_API void          ClosePopupToLevel(int remaining);
    IMGUI_API void          ClosePopupsOverWindow(ImGuiWindow* ref_window);
    IMGUI_API bool          IsPopupOpen(ImGuiID id);
    IMGUI_API bool          BeginPopupEx(ImGuiID id, ImGuiWindowFlags extra_flags);
    IMGUI_API void          BeginTooltipEx(ImGuiWindowFlags extra_flags, bool override_previous_tooltip = true);
    IMGUI_API ImGuiWindow*  GetFrontMostPopupModal();
    IMGUI_API ImVec2        FindBestWindowPosForPopup(ImGuiWindow* window);
    IMGUI_API ImVec2        FindBestWindowPosForPopupEx(const ImVec2& ref_pos, const ImVec2& size, ImGuiDir* last_dir, const ImRect& r_outer, const ImRect& r_avoid, ImGuiPopupPositionPolicy policy = ImGuiPopupPositionPolicy_Default);

    // Navigation
    IMGUI_API void          NavInitWindow(ImGuiWindow* window, bool force_reinit);
    IMGUI_API bool          NavMoveRequestButNoResultYet();
    IMGUI_API void          NavMoveRequestCancel();
    IMGUI_API void          NavMoveRequestForward(ImGuiDir move_dir, ImGuiDir clip_dir, const ImRect& bb_rel, ImGuiNavMoveFlags move_flags);
    IMGUI_API void          NavMoveRequestTryWrapping(ImGuiWindow* window, ImGuiNavMoveFlags move_flags);
    IMGUI_API float         GetNavInputAmount(ImGuiNavInput n, ImGuiInputReadMode mode);
    IMGUI_API ImVec2        GetNavInputAmount2d(ImGuiNavDirSourceFlags dir_sources, ImGuiInputReadMode mode, float slow_factor = 0.0f, float fast_factor = 0.0f);
    IMGUI_API int           CalcTypematicPressedRepeatAmount(float t, float t_prev, float repeat_delay, float repeat_rate);
    IMGUI_API void          ActivateItem(ImGuiID id);   // Remotely activate a button, checkbox, tree node etc. given its unique ID. activation is queued and processed on the next frame when the item is encountered again.
    IMGUI_API void          SetNavID(ImGuiID id, int nav_layer);
    IMGUI_API void          SetNavIDWithRectRel(ImGuiID id, int nav_layer, const ImRect& rect_rel);

    // Inputs
    inline bool             IsKeyPressedMap(ImGuiKey key, bool repeat = true)           { const int key_index = GImGui->IO.KeyMap[key]; return (key_index >= 0) ? IsKeyPressed(key_index, repeat) : false; }
    inline bool             IsNavInputDown(ImGuiNavInput n)                             { return GImGui->IO.NavInputs[n] > 0.0f; }
    inline bool             IsNavInputPressed(ImGuiNavInput n, ImGuiInputReadMode mode) { return GetNavInputAmount(n, mode) > 0.0f; }
    inline bool             IsNavInputPressedAnyOfTwo(ImGuiNavInput n1, ImGuiNavInput n2, ImGuiInputReadMode mode) { return (GetNavInputAmount(n1, mode) + GetNavInputAmount(n2, mode)) > 0.0f; }
    
    // Docking
    IMGUI_API void          DockContextInitialize(ImGuiContext* ctx);
    IMGUI_API void          DockContextShutdown(ImGuiContext* ctx);
    IMGUI_API void          DockContextOnLoadSettings(ImGuiContext* ctx);
    IMGUI_API void          DockContextRebuild(ImGuiContext* ctx);
    IMGUI_API void          DockContextNewFrameUpdateUndocking(ImGuiContext* ctx);
    IMGUI_API void          DockContextNewFrameUpdateDocking(ImGuiContext* ctx);
    IMGUI_API void          DockContextEndFrame(ImGuiContext* ctx);
    IMGUI_API void          DockContextQueueUndockWindow(ImGuiContext* ctx, ImGuiWindow* window);
    IMGUI_API void          DockContextQueueUndockNode(ImGuiContext* ctx, ImGuiDockNode* node);
    inline ImGuiDockNode*   DockNodeGetRootNode(ImGuiDockNode* node) { while (node->ParentNode) node = node->ParentNode; return node; }
    IMGUI_API void          BeginDocked(ImGuiWindow* window, bool* p_open);
    IMGUI_API void          BeginAsDockableDragDropSource(ImGuiWindow* window);
    IMGUI_API void          BeginAsDockableDragDropTarget(ImGuiWindow* window);
    IMGUI_API void          SetWindowDock(ImGuiWindow* window, ImGuiID dock_id, ImGuiCond cond);
    IMGUI_API void          ShowDockingDebug();

    // Docking - Builder function needs to be generally called before the DockSpace() node is submitted.
    IMGUI_API void          DockBuilderDockWindow(const char* window_name, ImGuiID node_id);
    IMGUI_API ImGuiDockNode*DockBuilderGetNode(ImGuiID node_id);
    IMGUI_API void          DockBuilderAddNode(ImGuiID node_id, ImVec2 ref_size, ImGuiDockNodeFlags flags = 0);
    IMGUI_API void          DockBuilderRemoveNode(ImGuiID node_id);                  // Remove node and all its child, undock all windows
    IMGUI_API void          DockBuilderRemoveNodeDockedWindows(ImGuiID node_id, bool clear_persistent_docking_references = true);
    IMGUI_API void          DockBuilderRemoveNodeChildNodes(ImGuiID node_id);        // Remove all split/hierarchy. All remaining docked windows will be re-docked to the root.
    IMGUI_API ImGuiID       DockBuilderSplitNode(ImGuiID node_id, ImGuiDir split_dir, float size_ratio_for_node_at_dir, ImGuiID* out_id_dir, ImGuiID* out_id_other);
    IMGUI_API void          DockBuilderCopyDockspace(ImGuiID src_dockspace_id, ImGuiID dst_dockspace_id, ImVector<const char*>* in_window_remap_pairs);
    IMGUI_API void          DockBuilderCopyNode(ImGuiID src_node_id, ImGuiID dst_node_id, ImVector<ImGuiID>* out_node_remap_pairs);
    IMGUI_API void          DockBuilderCopyWindowSettings(const char* src_name, const char* dst_name);
    IMGUI_API void          DockBuilderFinish(ImGuiID node_id);

    // Drag and Drop
    IMGUI_API bool          BeginDragDropTargetCustom(const ImRect& bb, ImGuiID id);
    IMGUI_API void          ClearDragDrop();
    IMGUI_API bool          IsDragDropPayloadBeingAccepted();

    // New Columns API (FIXME-WIP)
    IMGUI_API void          BeginColumns(const char* str_id, int count, ImGuiColumnsFlags flags = 0); // setup number of columns. use an identifier to distinguish multiple column sets. close with EndColumns().
    IMGUI_API void          EndColumns();                                                             // close columns
    IMGUI_API void          PushColumnClipRect(int column_index = -1);

    // Tab Bars
    IMGUI_API bool          BeginTabBarEx(ImGuiTabBar* tab_bar, const ImRect& bb, ImGuiTabBarFlags flags, ImGuiDockNode* dock_node);
    IMGUI_API ImGuiTabItem* TabBarFindTabByID(ImGuiTabBar* tab_bar, ImGuiID tab_id);
    IMGUI_API void          TabBarAddTab(ImGuiTabBar* tab_bar, ImGuiWindow* window, ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None);
    IMGUI_API void          TabBarRemoveTab(ImGuiTabBar* tab_bar, ImGuiID tab_id);
    IMGUI_API void          TabBarCloseTab(ImGuiTabBar* tab_bar, ImGuiTabItem* tab);
    IMGUI_API void          TabBarQueueChangeTabOrder(ImGuiTabBar* tab_bar, const ImGuiTabItem* tab, int dir);
    IMGUI_API bool          TabItemEx(ImGuiTabBar* tab_bar, const char* label, bool* p_open, ImGuiTabItemFlags flags, ImGuiWindow* docked_window);
    IMGUI_API ImVec2        TabItemCalcSize(const char* label, bool has_close_button);
    IMGUI_API void          TabItemBackground(ImDrawList* draw_list, const ImRect& bb, ImGuiTabItemFlags flags, ImU32 col);
    IMGUI_API bool          TabItemLabelAndCloseButton(ImDrawList* draw_list, const ImRect& bb, ImGuiTabItemFlags flags, const char* label, ImGuiID tab_id, ImGuiID close_button_id);

    // Render helpers
    // AVOID USING OUTSIDE OF IMGUI.CPP! NOT FOR PUBLIC CONSUMPTION. THOSE FUNCTIONS ARE A MESS. THEIR SIGNATURE AND BEHAVIOR WILL CHANGE, THEY NEED TO BE REFACTORED INTO SOMETHING DECENT.
    // NB: All position are in absolute pixels coordinates (we are never using window coordinates internally)
    IMGUI_API void          RenderText(ImVec2 pos, const char* text, const char* text_end = NULL, bool hide_text_after_hash = true);
    IMGUI_API void          RenderTextWrapped(ImVec2 pos, const char* text, const char* text_end, float wrap_width);
    IMGUI_API void          RenderTextClipped(const ImVec2& pos_min, const ImVec2& pos_max, const char* text, const char* text_end, const ImVec2* text_size_if_known, const ImVec2& align = ImVec2(0,0), const ImRect* clip_rect = NULL);
    IMGUI_API void          RenderTextClippedEx(ImDrawList* draw_list, const ImVec2& pos_min, const ImVec2& pos_max, const char* text, const char* text_end, const ImVec2* text_size_if_known, const ImVec2& align = ImVec2(0, 0), const ImRect* clip_rect = NULL);
    IMGUI_API void          RenderFrame(ImVec2 p_min, ImVec2 p_max, ImU32 fill_col, bool border = true, float rounding = 0.0f);
    IMGUI_API void          RenderFrameBorder(ImVec2 p_min, ImVec2 p_max, float rounding = 0.0f);
    IMGUI_API void          RenderColorRectWithAlphaCheckerboard(ImVec2 p_min, ImVec2 p_max, ImU32 fill_col, float grid_step, ImVec2 grid_off, float rounding = 0.0f, int rounding_corners_flags = ~0);
    IMGUI_API void          RenderArrow(ImVec2 pos, ImGuiDir dir, float scale = 1.0f);
    IMGUI_API void          RenderBullet(ImVec2 pos);
    IMGUI_API void          RenderCheckMark(ImVec2 pos, ImU32 col, float sz);
    IMGUI_API void          RenderNavHighlight(const ImRect& bb, ImGuiID id, ImGuiNavHighlightFlags flags = ImGuiNavHighlightFlags_TypeDefault); // Navigation highlight
    IMGUI_API void          RenderMouseCursor(ImVec2 pos, float scale, ImGuiMouseCursor mouse_cursor = ImGuiMouseCursor_Arrow);
    IMGUI_API const char*   FindRenderedTextEnd(const char* text, const char* text_end = NULL); // Find the optional ## from which we stop displaying text.
    IMGUI_API void          LogRenderedText(const ImVec2* ref_pos, const char* text, const char* text_end = NULL);

    // Render helpers (those functions don't access any ImGui state!)
    IMGUI_API void          RenderArrowPointingAt(ImDrawList* draw_list, ImVec2 pos, ImVec2 half_sz, ImGuiDir direction, ImU32 col);
    IMGUI_API void          RenderArrowDockMenu(ImDrawList* draw_list, ImVec2 p_min, float sz, ImU32 col);
    IMGUI_API void          RenderRectFilledRangeH(ImDrawList* draw_list, const ImRect& rect, ImU32 col, float x_start_norm, float x_end_norm, float rounding);
    IMGUI_API void          RenderRectFilledWithHole(ImDrawList* draw_list, ImRect outer, ImRect inner, ImU32 col, float rounding);
    IMGUI_API void          RenderPixelEllipsis(ImDrawList* draw_list, ImFont* font, ImVec2 pos, int count, ImU32 col);

    // Widgets
    IMGUI_API bool          ButtonEx(const char* label, const ImVec2& size_arg = ImVec2(0,0), ImGuiButtonFlags flags = 0);
    IMGUI_API bool          CloseButton(ImGuiID id, const ImVec2& pos, float radius);
    IMGUI_API bool          CollapseButton(ImGuiID id, const ImVec2& pos, ImGuiDockNode* dock_node);
    IMGUI_API bool          ArrowButtonEx(const char* str_id, ImGuiDir dir, ImVec2 size_arg, ImGuiButtonFlags flags);
    IMGUI_API void          Scrollbar(ImGuiLayoutType direction);
    IMGUI_API void          VerticalSeparator();        // Vertical separator, for menu bars (use current line height). Not exposed because it is misleading and it doesn't have an effect on regular layout.

    // Widgets low-level behaviors
    IMGUI_API bool          ButtonBehavior(const ImRect& bb, ImGuiID id, bool* out_hovered, bool* out_held, ImGuiButtonFlags flags = 0);
    IMGUI_API bool          DragBehavior(ImGuiID id, ImGuiDataType data_type, void* v, float v_speed, const void* v_min, const void* v_max, const char* format, float power);
    IMGUI_API bool          SliderBehavior(const ImRect& bb, ImGuiID id, ImGuiDataType data_type, void* v, const void* v_min, const void* v_max, const char* format, float power, ImGuiSliderFlags flags, ImRect* out_grab_bb);
    IMGUI_API bool          SplitterBehavior(const ImRect& bb, ImGuiID id, ImGuiAxis axis, float* size1, float* size2, float min_size1, float min_size2, float hover_extend = 0.0f, float hover_visibility_delay = 0.0f);
    IMGUI_API bool          TreeNodeBehavior(ImGuiID id, ImGuiTreeNodeFlags flags, const char* label, const char* label_end = NULL);
    IMGUI_API bool          TreeNodeBehaviorIsOpen(ImGuiID id, ImGuiTreeNodeFlags flags = 0);                     // Consume previous SetNextTreeNodeOpened() data, if any. May return true when logging
    IMGUI_API void          TreePushRawID(ImGuiID id);

    // Template functions are instantiated in imgui_widgets.cpp for a finite number of types. 
    // To use them externally (for custom widget) you may need an "extern template" statement in your code in order to link to existing instances and silence Clang warnings (see #2036).
    // e.g. " extern template IMGUI_API float RoundScalarWithFormatT<float, float>(const char* format, ImGuiDataType data_type, float v); "
    template<typename T, typename SIGNED_T, typename FLOAT_T>   IMGUI_API bool  DragBehaviorT(ImGuiDataType data_type, T* v, float v_speed, const T v_min, const T v_max, const char* format, float power);
    template<typename T, typename SIGNED_T, typename FLOAT_T>   IMGUI_API bool  SliderBehaviorT(const ImRect& bb, ImGuiID id, ImGuiDataType data_type, T* v, const T v_min, const T v_max, const char* format, float power, ImGuiSliderFlags flags, ImRect* out_grab_bb);
    template<typename T, typename FLOAT_T>                      IMGUI_API float SliderCalcRatioFromValueT(ImGuiDataType data_type, T v, T v_min, T v_max, float power, float linear_zero_pos);
    template<typename T, typename SIGNED_T>                     IMGUI_API T     RoundScalarWithFormatT(const char* format, ImGuiDataType data_type, T v);

    // InputText
    IMGUI_API bool          InputTextEx(const char* label, char* buf, int buf_size, const ImVec2& size_arg, ImGuiInputTextFlags flags, ImGuiInputTextCallback callback = NULL, void* user_data = NULL);
    IMGUI_API bool          InputScalarAsWidgetReplacement(const ImRect& bb, ImGuiID id, const char* label, ImGuiDataType data_type, void* data_ptr, const char* format);

    // Color
    IMGUI_API void          ColorTooltip(const char* text, const float* col, ImGuiColorEditFlags flags);
    IMGUI_API void          ColorEditOptionsPopup(const float* col, ImGuiColorEditFlags flags);
    IMGUI_API void          ColorPickerOptionsPopup(const float* ref_col, ImGuiColorEditFlags flags);

    // Plot
    IMGUI_API void          PlotEx(ImGuiPlotType plot_type, const char* label, float (*values_getter)(void* data, int idx), void* data, int values_count, int values_offset, const char* overlay_text, float scale_min, float scale_max, ImVec2 graph_size);

    // Shade functions (write over already created vertices)
    IMGUI_API void          ShadeVertsLinearColorGradientKeepAlpha(ImDrawList* draw_list, int vert_start_idx, int vert_end_idx, ImVec2 gradient_p0, ImVec2 gradient_p1, ImU32 col0, ImU32 col1);
    IMGUI_API void          ShadeVertsLinearUV(ImDrawList* draw_list, int vert_start_idx, int vert_end_idx, const ImVec2& a, const ImVec2& b, const ImVec2& uv_a, const ImVec2& uv_b, bool clamp);

} // namespace ImGui

// ImFontAtlas internals
IMGUI_API bool              ImFontAtlasBuildWithStbTruetype(ImFontAtlas* atlas);
IMGUI_API void              ImFontAtlasBuildRegisterDefaultCustomRects(ImFontAtlas* atlas);
IMGUI_API void              ImFontAtlasBuildSetupFont(ImFontAtlas* atlas, ImFont* font, ImFontConfig* font_config, float ascent, float descent);
IMGUI_API void              ImFontAtlasBuildPackCustomRects(ImFontAtlas* atlas, void* spc);
IMGUI_API void              ImFontAtlasBuildFinish(ImFontAtlas* atlas);
IMGUI_API void              ImFontAtlasBuildMultiplyCalcLookupTable(unsigned char out_table[256], float in_multiply_factor);
IMGUI_API void              ImFontAtlasBuildMultiplyRectAlpha8(const unsigned char table[256], unsigned char* pixels, int x, int y, int w, int h, int stride);

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning (pop)
#endif
