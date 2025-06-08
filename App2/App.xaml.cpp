#include "pch.h"
#include "App.xaml.h"
#include "MainWindow.xaml.h"
#include <winrt/Microsoft.UI.Windowing.h>
#include <winrt/Microsoft.UI.h>
#include <winrt/Windows.Graphics.h>
#include <Microsoft.UI.Xaml.Window.h>
#include <Windows.h>
#include <winrt/Microsoft.UI.Dispatching.h>
#include <winrt/Microsoft.UI.Xaml.Hosting.h>
#include <winrt/Windows.ApplicationModel.h>
#include <winrt/Windows.System.h>
#include <winrt/base.h>

#include <Unknwn.h>
#include <windows.graphics.h>


#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>

using namespace winrt;
using namespace Microsoft::UI::Xaml;
using namespace winrt::Microsoft::UI::Windowing;
using namespace winrt::Windows::Graphics;

// IWindowNative — для отримання HWND
//struct __declspec(uuid("6d5140c1-7436-11ce-8034-00aa006009fa")) IWindowNative : ::IUnknown
//{
//    virtual HRESULT __stdcall get_WindowHandle(HWND* hwnd) = 0;
//};


HWND GetWindowHandle(winrt::Microsoft::UI::Xaml::Window const& window)
{
    HWND hwnd{ nullptr };
    auto windowNative = window.as<IWindowNative>();
    windowNative->get_WindowHandle(&hwnd);
    return hwnd;
}



namespace winrt::App2::implementation
{
    App::App()
    {
    #if defined _DEBUG && !defined DISABLE_XAML_GENERATED_BREAK_ON_UNHANDLED_EXCEPTION
        UnhandledException([](IInspectable const&, UnhandledExceptionEventArgs const& e)
            {
                if (IsDebuggerPresent())
                {
                    auto errorMessage = e.Message();
                    __debugbreak();
                }
            });
    #endif
    }

    void App::OnLaunched([[maybe_unused]] LaunchActivatedEventArgs const& e)
    {
        window = make<MainWindow>();
        window.Activate();

        HWND hwnd = GetWindowHandle(window);

        // Задаємо фіксований розмір
        SetWindowPos(hwnd, nullptr, 0, 0, 800, 750, SWP_NOMOVE | SWP_NOZORDER); // размер окна

        // Вимикаємо зміну розміру
        LONG style = GetWindowLong(hwnd, GWL_STYLE);
        style &= ~(WS_THICKFRAME | WS_MAXIMIZEBOX); // прибирає можливість змінювати розмір і згортати
        SetWindowLong(hwnd, GWL_STYLE, style);
    }
}
