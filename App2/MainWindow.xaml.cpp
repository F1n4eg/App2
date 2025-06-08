#include "pch.h"
#include "Algorithms.h" 

using std::numeric_limits;
using std::streamsize;

#pragma comment(lib, "Shell32.lib")

using namespace winrt;
using namespace Microsoft::UI::Xaml;
using namespace std;

namespace winrt::App2::implementation
{
    winrt::Microsoft::UI::Xaml::Window settingsWindow{ nullptr };

    int32_t MainWindow::MyProperty()
    {

        return 0;
    }

    void MainWindow::MyProperty(int32_t)
    {

    }

    void MainWindow::OnCuckooSearchClick(IInspectable const&, RoutedEventArgs const&)
    {
        int runCount = 1;
        try {
            runCount = std::stoi(winrt::to_string(runCountBox().Text()));
        }
        catch (...) {
            runCount = 1;
        }
        if (runCount < 1) runCount = 1;
        if (runCount > 10) runCount = 10;

        std::ostringstream buffer;
        auto* old_buf = std::cout.rdbuf(buffer.rdbuf());

        for (int i = 0; i < runCount; ++i) {
            CuckooSearch cuckoo(10, 20, 100000);
            buffer << "Run " << (i + 1) << ":\n";
            cuckoo.search();
            buffer << "\n";
        }

        std::cout.rdbuf(old_buf);
        result().Text(to_hstring(buffer.str()));
    }

    void MainWindow::OnPSOClick(IInspectable const&  , RoutedEventArgs const&  )
    {
        int runCount = 1;
        try {
            runCount = std::stoi(winrt::to_string(runCountBox().Text()));
        }
        catch (...) {
            runCount = 1;
        }
        if (runCount < 1) runCount = 1;
        if (runCount > 10) runCount = 10;

        std::ostringstream buffer;
        auto* old_buf = std::cout.rdbuf(buffer.rdbuf());

        size_t maxIter = 10000;
        size_t swarmSize = 100;
        double cognitiveAttraction = 0.8;
        double socialAttraction = 1.2;

        for (int i = 0; i < runCount; ++i) {
            buffer << "Run " << (i + 1) << ":\n";
            PSO pso(maxIter, swarmSize, cognitiveAttraction, socialAttraction);
            pso.search();
            buffer << "\n";
        }

        std::cout.rdbuf(old_buf);
        result().Text(to_hstring(buffer.str()));
    }

    void MainWindow::OnSimulatedAnnealingClick(IInspectable const&  , RoutedEventArgs const&  )
    {
        int runCount = 1;
        try {
            runCount = std::stoi(winrt::to_string(runCountBox().Text()));
        }
        catch (...) {
            runCount = 1;
        }
        if (runCount < 1) runCount = 1;
        if (runCount > 10) runCount = 10;

        std::ostringstream buffer;
        auto* old_buf = std::cout.rdbuf(buffer.rdbuf());

        for (int i = 0; i < runCount; ++i) {
            buffer << "Run " << (i + 1) << ":\n";
            SimulatedAnnealing sa(100, 0.5, 10000000);
            sa.search();
            buffer << "\n";
        }

        std::cout.rdbuf(old_buf);
        result().Text(to_hstring(buffer.str()));
    }

    void MainWindow::OnPatternSearchClick(IInspectable const&  , RoutedEventArgs const&  )
    {
        int runCount = 1;
        try {
            runCount = std::stoi(winrt::to_string(runCountBox().Text()));
        }
        catch (...) {
            runCount = 1;
        }
        if (runCount < 1) runCount = 1;
        if (runCount > 10) runCount = 10;

        std::ostringstream buffer;
        auto* old_buf = std::cout.rdbuf(buffer.rdbuf());

        double stepSize = 5.0;
        double tolerance = 1e-5;
        int maxIterations = 10000;

        for (int i = 0; i < runCount; ++i) {
            buffer << "Run " << (i + 1) << ":\n";
           /* std::vector<double> startPoint = randomSolution();
            PatternSearch ps(startPoint, stepSize, tolerance, maxIterations);
            ps.search();*/
            vector<double> start = randomSolution();
            vector<double> initDelta = { 0.125, 0.125, 5.0, 5.0 }; // початкові кроки
            HookeJeeves hj(start, initDelta, 1e-6, 2.0, 10000);
            hj.search();
            buffer << "\n";
        }

        std::cout.rdbuf(old_buf);
        result().Text(to_hstring(buffer.str()));
    }

    void MainWindow::OnFireflyAlgorithmClick(IInspectable const&  , RoutedEventArgs const&  )
    {
        int runCount = 1;
        try {
            runCount = std::stoi(winrt::to_string(runCountBox().Text()));
        }
        catch (...) {
            runCount = 1;
        }
        if (runCount < 1) runCount = 1;
        if (runCount > 10) runCount = 10;

        std::ostringstream buffer;
        auto* old_buf = std::cout.rdbuf(buffer.rdbuf());

        for (int i = 0; i < runCount; ++i) {
            buffer << "Run " << (i + 1) << ":\n";
            FireflyAlgorithm ff(30, 1000, 0.25, 1.0, 1.0);
            ff.run();
            buffer << "\n";
        }

        std::cout.rdbuf(old_buf);
        result().Text(to_hstring(buffer.str()));
    }

    void MainWindow::FlowerPolinationClick(IInspectable const&  , RoutedEventArgs const&  )
    {
        int runCount = 1;
        try {
            runCount = std::stoi(winrt::to_string(runCountBox().Text()));
        }
        catch (...) {
            runCount = 1;
        }
        if (runCount < 1) runCount = 1;
        if (runCount > 10) runCount = 10;

        std::ostringstream buffer;
        auto* old_buf = std::cout.rdbuf(buffer.rdbuf());

        for (int i = 0; i < runCount; ++i) {
            buffer << "Run " << (i + 1) << ":\n";
            FlowerPollinationAlgorithm fpa(100, 10000);
            fpa.search();
            buffer << "\n";
        }

        std::cout.rdbuf(old_buf);
        result().Text(to_hstring(buffer.str()));
    }

    void MainWindow::GravitationalsearchClick(IInspectable const&  , RoutedEventArgs const&)
    {
        int runCount = 1;
        try {
            runCount = std::stoi(winrt::to_string(runCountBox().Text()));
        }
        catch (...) {
            runCount = 1;
        }
        if (runCount < 1) runCount = 1;
        if (runCount > 10) runCount = 10;

        std::ostringstream buffer;
        auto* old_buf = std::cout.rdbuf(buffer.rdbuf());

        for (int i = 0; i < runCount; ++i) {
            buffer << "Run " << (i + 1) << ":\n";
            GravitationalSearchAlgorithm gsa(40, 20000, 100);
            gsa.search();
            buffer << "\n";
        }

        std::cout.rdbuf(old_buf);
        result().Text(to_hstring(buffer.str()));
    }

    void winrt::App2::implementation::MainWindow::OnInfoClick(IInspectable const&, RoutedEventArgs const&)
    {
        using namespace winrt::Microsoft::UI;
        using namespace winrt::Microsoft::UI::Windowing;
        using namespace winrt::Microsoft::UI::Xaml;
        using namespace winrt::Microsoft::UI::Xaml::Controls;
        using namespace winrt::Microsoft::UI::Xaml::Media::Imaging;
        using namespace Windows::Graphics;

        // Статическая переменная для хранения состояния окна
        static Window infoWindow{ nullptr };

        // Проверяем, если окно уже открыто, активируем его и выходим
        if (infoWindow)
        {
            infoWindow.Activate();
            return;
        }

        // Создаем новое окно
        infoWindow = Window{};

        // Загружаем изображение
        Image image{};
        BitmapImage bitmap{};
        bitmap.UriSource(winrt::Windows::Foundation::Uri{ L"C:/Users/Sasha/Desktop/Untitled.png" });
        image.Source(bitmap);
        image.Stretch(Media::Stretch::UniformToFill);

        // Устанавливаем содержимое окна
        infoWindow.Content(image);

        // Получаем AppWindow и задаем размер
        HWND hwnd = GetWindowHandle(infoWindow);
        auto appWindow = AppWindow::GetFromWindowId(WindowId{ reinterpret_cast<uint64_t>(hwnd) });
        appWindow.Resize(SizeInt32{ 800, 600 });

        // Обработчик закрытия окна, чтобы сбросить состояние
        infoWindow.Closed([&](auto const&, auto const&) {
            infoWindow = nullptr;
            });

        // Показываем окно
        infoWindow.Activate();
    }

    void ExportToExcel(const std::vector<std::vector<std::string>>& data, const std::string& filename) {
        lxw_workbook* workbook = workbook_new(filename.c_str());
        lxw_worksheet* worksheet = workbook_add_worksheet(workbook, nullptr);

        for (size_t row = 0; row < data.size(); ++row) {
            for (size_t col = 0; col < data[row].size(); ++col) {
                const std::string& cell = data[row][col];
                if (row == 0) {
                    // Заголовки всегда строка
                    worksheet_write_string(worksheet, row, col, cell.c_str(), nullptr);
                }
                else {
                    char* endptr = nullptr;
                    double num = std::strtod(cell.c_str(), &endptr);
                    if (endptr != cell.c_str() && *endptr == '\0') {
                        worksheet_write_number(worksheet, row, col, num, nullptr);
                    }
                    else {
                        worksheet_write_string(worksheet, row, col, cell.c_str(), nullptr);
                    }
                }
            }
        }
        workbook_close(workbook);
    }

    void winrt::App2::implementation::MainWindow::Excell_Click(IInspectable const&, RoutedEventArgs const&)
    {
        std::string text = winrt::to_string(result().Text());

        std::vector<std::vector<std::string>> data;
        // Заголовок таблиці
        data.push_back({ "Run", "Time (s)", "Cost", "x1", "x2", "x3", "x4" });

        std::regex blockRegex(R"(Run (\d+):\s*Execution Time:\s*([\d.]+)\s*seconds\s*Best Fitness:\s*([\d.]+)\s*Best Solution:\s*\(([^)]+)\))");
        std::smatch match;
        auto text_begin = text.cbegin();
        auto text_end = text.cend();

        while (std::regex_search(text_begin, text_end, match, blockRegex)) {
            std::string run = match[1];
            std::string time = match[2];
            std::string fitness = match[3];
            std::string solution = match[4];

            // НЕ заменяем точку на запятую!
            std::vector<std::string> row = { run, time, fitness };

            std::stringstream ss(solution);
            std::string value;
            while (std::getline(ss, value, ',')) {
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                // НЕ заменяем точку на запятую!
                row.push_back(value);
            }

            data.push_back(row);
            text_begin = match.suffix().first;
        }



        ExportToExcel(data, "C:/Users/Sasha/Desktop/results.xlsx");

    }

    void winrt::App2::implementation::MainWindow::OnLengthRadioChecked(IInspectable const& sender, RoutedEventArgs const&)
    {
        auto radio = sender.as<winrt::Microsoft::UI::Xaml::Controls::RadioButton>();
        std::string value = winrt::to_string(radio.Content().as<hstring>());

        // Извлекаем только число из строки
        std::smatch match;
        int number = 0;
        if (std::regex_search(value, match, std::regex(R"((\d+))"))) {
            number = std::stoi(match[1]);
        }

        // Теперь сравниваем только число
        if (number == 240) {
            X4_MAX = 240.0;
        }
        else if (number == 200) {
            X4_MAX = 200.0;
        }
    }

    void MainWindow::OnSettingsClick(IInspectable const&, RoutedEventArgs const&)
    {
        using namespace winrt::Microsoft::UI;
        using namespace winrt::Microsoft::UI::Windowing;
        using namespace winrt::Microsoft::UI::Xaml;
        using namespace winrt::Microsoft::UI::Xaml::Controls;
        using namespace Windows::Graphics;
        using namespace std;
        using namespace winrt::Microsoft::UI::Dispatching;
        using namespace winrt::Microsoft::UI::Xaml::Media;
        using namespace winrt::Windows::UI;
        using namespace winrt::Windows::Foundation;

        if (settingsWindow)
        {
            settingsWindow.Activate();
            return;
        }


        settingsWindow = Window{};

        StackPanel panel{};
        panel.Orientation(Orientation::Vertical);
        panel.Padding(ThicknessHelper::FromUniformLength(20));
        panel.Background(
            winrt::Microsoft::UI::Xaml::Media::SolidColorBrush{
                winrt::Windows::UI::ColorHelper::FromArgb(
                    0xFF,
                    0x22,
                    0x28,
                    0x31
                )
            }
        );




        TextBlock title{};
        title.Text(L"Налаштування алгоритмів");
        title.FontSize(20);
        title.Margin(ThicknessHelper::FromUniformLength(10));
        panel.Children().Append(title);

        Button btnSave{};
        btnSave.Content(box_value(L"Зберегти налаштування"));
        btnSave.Margin(ThicknessHelper::FromUniformLength(10));
        panel.Children().Append(btnSave);

        RadioButton rbDefault{};
        rbDefault.Content(box_value(L"Використовувати стандартні налаштування"));
        rbDefault.IsChecked(true);
        rbDefault.Margin(ThicknessHelper::FromUniformLength(5));
        panel.Children().Append(rbDefault);

        RadioButton rbManual{};
        rbManual.Content(box_value(L"Налаштувати вручну"));
        rbManual.Margin(ThicknessHelper::FromUniformLength(5));
        panel.Children().Append(rbManual);

        TextBox tbD{};    tbD.PlaceholderText(L"D (щільність сталі)");             
        tbD.Visibility(Visibility::Collapsed); 
        tbD.Margin(ThicknessHelper::FromLengths(0, 0, 0, 5));
        panel.Children().Append(tbD);

        TextBox tbCw{};   tbCw.PlaceholderText(L"Cw (вартість зварювання)");   
        tbCw.Visibility(Visibility::Collapsed); 
        tbCw.Margin(ThicknessHelper::FromLengths(0, 0, 0, 5)); 
        panel.Children().Append(tbCw);

        TextBox tbCs{};   tbCs.PlaceholderText(L"Cs (вартість оболонки)"); 
        tbCs.Visibility(Visibility::Collapsed); 
        tbCs.Margin(ThicknessHelper::FromLengths(0, 0, 0, 5));
        panel.Children().Append(tbCs);

        TextBox tbCh{};  
        tbCh.PlaceholderText(L"Ch (вартість кришки)");   
        tbCh.Visibility(Visibility::Collapsed);
        tbCh.Margin(ThicknessHelper::FromLengths(0, 0, 0, 5));
        panel.Children().Append(tbCh);


        TextBox Vmin{};
        Vmin.PlaceholderText(L"Мінімальний об'єм баку");
        Vmin.Visibility(Visibility::Collapsed);
        Vmin.Margin(ThicknessHelper::FromLengths(0, 0, 0, 15));
        panel.Children().Append(Vmin);



        // Показуємо поля при виборі ручних налаштувань
        rbManual.Checked([tbD, tbCw, tbCs, tbCh, Vmin](auto const&, auto const&) {
            tbD.Visibility(Visibility::Visible);
            tbCw.Visibility(Visibility::Visible);
            tbCs.Visibility(Visibility::Visible);
            tbCh.Visibility(Visibility::Visible);

			Vmin.Visibility(Visibility::Visible);
            });

        // Ховаємо поля при стандартних налаштуваннях
        rbDefault.Checked([tbD, tbCw, tbCs, tbCh, Vmin](auto const&, auto const&) {
            tbD.Visibility(Visibility::Collapsed);
            tbCw.Visibility(Visibility::Collapsed);
            tbCs.Visibility(Visibility::Collapsed);
            tbCh.Visibility(Visibility::Collapsed);
			
			Vmin.Visibility(Visibility::Collapsed);
            });


        auto Blink = [&](Controls::TextBox const& tb)
            {
                // Зберігаємо оригінальну рамку
                auto orig = tb.BorderBrush();

                // Встановлюємо червону рамку

                tb.BorderBrush(SolidColorBrush(Windows::UI::Colors::Red()));

                // Створюємо таймер у DispatcherQueue
                DispatcherQueueTimer timer = DispatcherQueue::GetForCurrentThread().CreateTimer();
                timer.Interval(TimeSpan{ 5000000 }); // 500 ms
                // у Tick повертаємо оригінальну рамку і зупиняємо таймер
                timer.Tick([tb, orig, timer](auto&&, auto&&) mutable {
                    tb.BorderBrush(orig);
                    timer.Stop();
                    });
                timer.Start();
            };


        // Обробник кнопки "Сохранить настройки"
        btnSave.Click([tbD, tbCw, tbCs, tbCh, Vmin, rbDefault, Blink](auto const&, auto const&) {
            // Якщо обрано стандартні налаштування — закриваємо лише вікно налаштувань
            auto defaultChecked = rbDefault.IsChecked();
            if (defaultChecked && defaultChecked.Value())
            {
                flag = true;
                settingsWindow.Close();
                return;
            }
            bool hasError = false;

            // Перевіряємо порожні
            if (tbD.Text().empty()) { Blink(tbD);    hasError = true; }
            if (tbCw.Text().empty()) { Blink(tbCw);   hasError = true; }
            if (tbCs.Text().empty()) { Blink(tbCs);   hasError = true; }
            if (tbCh.Text().empty()) { Blink(tbCh);   hasError = true; }
            
            if (Vmin.Text().empty()) { Blink(Vmin);   hasError = true; }

            if (hasError)
                return;  // хоч раз мерехнуло — летимо назад



            flag = false;

            // Інакше читаємо текст із полів
            wstring sD = tbD.Text().c_str();
            wstring sCw = tbCw.Text().c_str();
            wstring sCs = tbCs.Text().c_str();
            wstring sCh = tbCh.Text().c_str();
           
            wstring sVmin = Vmin.Text().c_str();

            // Перевіряємо, що жодне поле не пусте
            if (sD.empty() || sCw.empty() || sCs.empty() || sCh.empty() || sVmin.empty())
                return;

            double vD, vCw, vCs, vCh,vVmax,vVmin;
            try
            {
                // Конвертуємо в double
                vD = stod(sD);
                vCw = stod(sCw);
                vCs = stod(sCs);
                vCh = stod(sCh);
			
				vVmin = stod(sVmin);

            }
            catch (...)
            {
                // Некоректний формат числа
                return;
            }

            // Перевіряємо, що жодне значення не нуль
            if (vD == 0.0 || vCw == 0.0 || vCs == 0.0 || vCh == 0.0 || vVmin == 0.0 )
                return;

            // Записуємо у статичні змінні
            D = vD;
            Cw = vCw;
            Cs = vCs;
            Ch = vCh;
     
            ValueMin = vVmin;


            calculateCoefficients(k1, k2, k3, k4);

            // Закриваємо тільки вікно налаштувань
            settingsWindow.Close();
            });

        // --- Фініш: вміст і розмір ---
        settingsWindow.Content(panel);

        HWND hwnd = GetWindowHandle(settingsWindow);
       // auto appWindow = AppWindow::GetFromWindowId(WindowId{reinterpret_cast<uint64_t>(hwnd) });


        // Задаємо фіксований розмір
        SetWindowPos(hwnd, nullptr, 0, 0, 450, 500, SWP_NOMOVE | SWP_NOZORDER); // размер окна
        // Вимикаємо зміну розміру
        LONG style = GetWindowLong(hwnd, GWL_STYLE);
        style &= ~(WS_THICKFRAME | WS_MAXIMIZEBOX); // прибирає можливість змінювати розмір і згортати
        SetWindowLong(hwnd, GWL_STYLE, style);



        // Скидаємо дескриптор при закритті
        settingsWindow.Closed([&](auto const&, auto const&) {
            settingsWindow = nullptr;
            });

        // Відкриваємо вікно
        settingsWindow.Activate();
    }

}



