#pragma once

#include "MainWindow.g.h"

#include <string>
#include <Windows.h>
#include <winrt/Microsoft.UI.Xaml.h>

HWND GetWindowHandle(winrt::Microsoft::UI::Xaml::Window const& window);

namespace winrt::App2::implementation
{

    extern winrt::Microsoft::UI::Xaml::Window settingsWindow;

    struct MainWindow : MainWindowT<MainWindow>
    {

        MainWindow()
        {
            // Xaml objects should not call InitializeComponent during construction.
            // See https://github.com/microsoft/cppwinrt/tree/master/nuget#initializecomponent
            this->Closed([&](auto const&, auto const&) {
                if (settingsWindow)
                {
                    settingsWindow.Close();
                    settingsWindow = nullptr;
                }
                });
        }

        int32_t MyProperty();
        void MyProperty(int32_t value);

        void OnCuckooSearchClick(IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void OnPSOClick(IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void OnSimulatedAnnealingClick(IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void OnPatternSearchClick(IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void OnFireflyAlgorithmClick(IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void FlowerPolinationClick(IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void GravitationalsearchClick(IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void OnLengthRadioChecked(winrt::Windows::Foundation::IInspectable const& sender, winrt::Microsoft::UI::Xaml::RoutedEventArgs const& e);
        void OnInfoClick(IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
		void OnSettingsClick(IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Excell_Click(winrt::Windows::Foundation::IInspectable const& sender, winrt::Microsoft::UI::Xaml::RoutedEventArgs const& e);
        
    };
}

namespace winrt::App2::factory_implementation
{
    struct MainWindow : MainWindowT<MainWindow, implementation::MainWindow>
    {
    };
}
