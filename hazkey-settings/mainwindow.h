#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QAbstractButton>
#include <QNetworkAccessManager>
#include <QWidget>
#include <memory>

#include "controllers/about_tab_controller.h"
#include "controllers/ai_tab_controller.h"
#include "controllers/conversion_tab_controller.h"
#include "controllers/dictionary_tab_controller.h"
#include "controllers/input_style_tab_controller.h"
#include "controllers/tab_context.h"
#include "controllers/user_interface_tab_controller.h"
#include "serverconnector.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QWidget {
    Q_OBJECT

   public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

   private slots:
    void onButtonClicked(QAbstractButton* button);
    void onApply();
    void onResetConfiguration();

   private:
    void connectSignals();
    bool loadCurrentConfig(bool fetchConfig = true);
    bool saveCurrentConfig();
    void bindContexts();
    void setupControllers();
    Ui::MainWindow* ui_;
    ServerConnector server_;
    hazkey::config::CurrentConfig currentConfig_;
    hazkey::config::Profile* currentProfile_;
    std::unique_ptr<QNetworkAccessManager> networkManager_;
    hazkey::settings::TabContext controllerContext_;
    std::unique_ptr<hazkey::settings::UserInterfaceTabController> uiTab_;
    std::unique_ptr<hazkey::settings::ConversionTabController> conversionTab_;
    std::unique_ptr<hazkey::settings::InputStyleTabController> inputStyleTab_;
    std::unique_ptr<hazkey::settings::DictionaryTabController> dictionaryTab_;
    std::unique_ptr<hazkey::settings::AiTabController> aiTab_;
    std::unique_ptr<hazkey::settings::AboutTabController> aboutTab_;
};
#endif  // MAINWINDOW_H
