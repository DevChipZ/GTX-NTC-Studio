#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <string>

// Real System32 dxgi.dll handle & function pointers
static HMODULE g_realDxgi = nullptr;
static std::once_flag g_initFlag;
static std::once_flag g_hookFlag;

using PFN_CreateDXGIFactory  = HRESULT(WINAPI*)(REFIID, void**);
using PFN_CreateDXGIFactory1 = HRESULT(WINAPI*)(REFIID, void**);
using PFN_CreateDXGIFactory2 = HRESULT(WINAPI*)(UINT, REFIID, void**);
using PFN_DXGIGetDebugInterface1 = HRESULT(WINAPI*)(UINT, REFIID, void**);
using PFN_DXGIDeclareAdapterRemovalSupport = HRESULT(WINAPI*)();
using PFN_QueryVideoMemoryInfo = HRESULT(STDMETHODCALLTYPE*)(
    IDXGIAdapter3*, UINT, DXGI_MEMORY_SEGMENT_GROUP, DXGI_QUERY_VIDEO_MEMORY_INFO*);

static PFN_CreateDXGIFactory  g_origCreateFactory  = nullptr;
static PFN_CreateDXGIFactory1 g_origCreateFactory1 = nullptr;
static PFN_CreateDXGIFactory2 g_origCreateFactory2 = nullptr;
static PFN_DXGIGetDebugInterface1 g_origGetDebug1  = nullptr;
static PFN_DXGIDeclareAdapterRemovalSupport g_origDeclareRemoval = nullptr;
static PFN_QueryVideoMemoryInfo g_origQueryVram = nullptr;

// Configurable VRAM Guard settings (loaded from gtx_ntc_live.ini)
static uint64_t g_vramBudgetMB = 3200; // Safe ceiling for 4GB GTX 1650
static bool g_enableLog = true;
static std::string g_logPath;
static std::mutex g_logMutex;
static ULONGLONG g_lastLogTick = 0;

static void WriteLog(const char* fmt, ...) {
    if (!g_enableLog || g_logPath.empty()) return;
    std::lock_guard<std::mutex> lock(g_logMutex);
    FILE* f = nullptr;
    if (fopen_s(&f, g_logPath.c_str(), "a") == 0 && f) {
        va_list args;
        va_start(args, fmt);
        vfprintf(f, fmt, args);
        va_end(args);
        fprintf(f, "\n");
        fclose(f);
    }
}

static void LoadConfigAndRealDXGI() {
    std::call_once(g_initFlag, []() {
        // Resolve game directory for INI and Log
        char exePath[MAX_PATH] = {};
        GetModuleFileNameA(nullptr, exePath, MAX_PATH);
        std::string dir(exePath);
        size_t slash = dir.find_last_of("\\/");
        dir = (slash != std::string::npos) ? dir.substr(0, slash + 1) : "";

        std::string iniPath = dir + "gtx_ntc_live.ini";
        g_logPath = dir + "gtx_ntc_live.log";

        // Auto-create default config if missing
        DWORD attr = GetFileAttributesA(iniPath.c_str());
        if (attr == INVALID_FILE_ATTRIBUTES) {
            FILE* f = nullptr;
            if (fopen_s(&f, iniPath.c_str(), "w") == 0 && f) {
                fprintf(f, "[GTX_NTC_Live]\nVRAMBudgetMB=3200\nEnableLogging=1\n");
                fclose(f);
            }
        }

        g_vramBudgetMB = GetPrivateProfileIntA("GTX_NTC_Live", "VRAMBudgetMB", 3200, iniPath.c_str());
        g_enableLog = GetPrivateProfileIntA("GTX_NTC_Live", "EnableLogging", 1, iniPath.c_str()) != 0;

        // Clear old log on game launch
        if (g_enableLog) {
            FILE* f = nullptr;
            if (fopen_s(&f, g_logPath.c_str(), "w") == 0 && f) fclose(f);
        }

        WriteLog("[GTX-NTC Live] Loaded. Target 4GB-Safe VRAM Budget: %llu MB", g_vramBudgetMB);

        // Load real Windows System32\dxgi.dll
        wchar_t sysDir[MAX_PATH] = {};
        GetSystemDirectoryW(sysDir, MAX_PATH);
        std::wstring realPath = std::wstring(sysDir) + L"\\dxgi.dll";
        g_realDxgi = LoadLibraryW(realPath.c_str());

        if (g_realDxgi) {
            g_origCreateFactory  = reinterpret_cast<PFN_CreateDXGIFactory>(GetProcAddress(g_realDxgi, "CreateDXGIFactory"));
            g_origCreateFactory1 = reinterpret_cast<PFN_CreateDXGIFactory1>(GetProcAddress(g_realDxgi, "CreateDXGIFactory1"));
            g_origCreateFactory2 = reinterpret_cast<PFN_CreateDXGIFactory2>(GetProcAddress(g_realDxgi, "CreateDXGIFactory2"));
            g_origGetDebug1      = reinterpret_cast<PFN_DXGIGetDebugInterface1>(GetProcAddress(g_realDxgi, "DXGIGetDebugInterface1"));
            g_origDeclareRemoval = reinterpret_cast<PFN_DXGIDeclareAdapterRemovalSupport>(GetProcAddress(g_realDxgi, "DXGIDeclareAdapterRemovalSupport"));
        }
    });
}

static HRESULT STDMETHODCALLTYPE Hooked_QueryVideoMemoryInfo(
    IDXGIAdapter3* This,
    UINT NodeIndex,
    DXGI_MEMORY_SEGMENT_GROUP MemorySegmentGroup,
    DXGI_QUERY_VIDEO_MEMORY_INFO* pVideoMemoryInfo)
{
    HRESULT hr = g_origQueryVram(This, NodeIndex, MemorySegmentGroup, pVideoMemoryInfo);
    if (SUCCEEDED(hr) && pVideoMemoryInfo && MemorySegmentGroup == DXGI_MEMORY_SEGMENT_GROUP_LOCAL) {
        const uint64_t origBudgetMB = pVideoMemoryInfo->Budget / (1024ULL * 1024ULL);
        const uint64_t capBytes = g_vramBudgetMB * 1024ULL * 1024ULL;

        if (pVideoMemoryInfo->Budget > capBytes) {
            pVideoMemoryInfo->Budget = capBytes;
        }

        ULONGLONG now = GetTickCount64();
        if (now - g_lastLogTick >= 5000) {
            g_lastLogTick = now;
            uint64_t usedMB = pVideoMemoryInfo->CurrentUsage / (1024ULL * 1024ULL);
            uint64_t activeBudgetMB = pVideoMemoryInfo->Budget / (1024ULL * 1024ULL);
            WriteLog("[Telemetry] Live VRAM: %llu MB / Enforced Budget: %llu MB (OS Raw: %llu MB)",
                     usedMB, activeBudgetMB, origBudgetMB);
        }
    }
    return hr;
}

static void InstallVramGuardHook(void* factoryPtr) {
    if (!factoryPtr) return;
    std::call_once(g_hookFlag, [factoryPtr]() {
        IUnknown* unk = static_cast<IUnknown*>(factoryPtr);
        IDXGIFactory1* factory1 = nullptr;
        if (SUCCEEDED(unk->QueryInterface(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory1))) && factory1) {
            IDXGIAdapter1* adapter1 = nullptr;
            if (SUCCEEDED(factory1->EnumAdapters1(0, &adapter1)) && adapter1) {
                IDXGIAdapter3* adapter3 = nullptr;
                if (SUCCEEDED(adapter1->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3))) && adapter3) {
                    void** vtable = *reinterpret_cast<void***>(adapter3);
                    DWORD oldProtect = 0;
                    // IDXGIAdapter3::QueryVideoMemoryInfo is vtable index 14
                    if (VirtualProtect(&vtable[14], sizeof(void*), PAGE_EXECUTE_READWRITE, &oldProtect)) {
                        g_origQueryVram = reinterpret_cast<PFN_QueryVideoMemoryInfo>(vtable[14]);
                        vtable[14] = reinterpret_cast<void*>(&Hooked_QueryVideoMemoryInfo);
                        VirtualProtect(&vtable[14], sizeof(void*), oldProtect, &oldProtect);
                        WriteLog("[GTX-NTC Live] IDXGIAdapter3::QueryVideoMemoryInfo VRAM Guard active.");
                    }
                    adapter3->Release();
                }
                adapter1->Release();
            }
            factory1->Release();
        }
    });
}

extern "C" {

HRESULT WINAPI Proxy_CreateDXGIFactory(REFIID riid, void** ppFactory) {
    LoadConfigAndRealDXGI();
    if (!g_origCreateFactory) return E_FAIL;
    HRESULT hr = g_origCreateFactory(riid, ppFactory);
    if (SUCCEEDED(hr) && ppFactory) InstallVramGuardHook(*ppFactory);
    return hr;
}

HRESULT WINAPI Proxy_CreateDXGIFactory1(REFIID riid, void** ppFactory) {
    LoadConfigAndRealDXGI();
    if (!g_origCreateFactory1) return E_FAIL;
    HRESULT hr = g_origCreateFactory1(riid, ppFactory);
    if (SUCCEEDED(hr) && ppFactory) InstallVramGuardHook(*ppFactory);
    return hr;
}

HRESULT WINAPI Proxy_CreateDXGIFactory2(UINT Flags, REFIID riid, void** ppFactory) {
    LoadConfigAndRealDXGI();
    if (!g_origCreateFactory2) return E_FAIL;
    HRESULT hr = g_origCreateFactory2(Flags, riid, ppFactory);
    if (SUCCEEDED(hr) && ppFactory) InstallVramGuardHook(*ppFactory);
    return hr;
}

HRESULT WINAPI Proxy_DXGIGetDebugInterface1(UINT Flags, REFIID riid, void** pDebug) {
    LoadConfigAndRealDXGI();
    return g_origGetDebug1 ? g_origGetDebug1(Flags, riid, pDebug) : E_FAIL;
}

HRESULT WINAPI Proxy_DXGIDeclareAdapterRemovalSupport() {
    LoadConfigAndRealDXGI();
    return g_origDeclareRemoval ? g_origDeclareRemoval() : S_OK;
}

} // extern "C"

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID) {
    if (fdwReason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hinstDLL);
    }
    return TRUE;
}
