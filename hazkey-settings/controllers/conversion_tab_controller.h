#ifndef HAZKEY_SETTINGS_CONTROLLERS_CONVERSION_TAB_CONTROLLER_H_
#define HAZKEY_SETTINGS_CONTROLLERS_CONVERSION_TAB_CONTROLLER_H_

#include <QObject>

#include "controllers/tab_context.h"

class QWidget;

namespace Ui {
class MainWindow;
}

namespace hazkey::settings {

class ConversionTabController : public QObject {
    Q_OBJECT

   public:
    ConversionTabController(Ui::MainWindow* ui, QWidget* window,
                            ServerConnector* server, QObject* parent);
    void setContext(const TabContext& context);
    void connectSignals();
    void loadFromConfig();
    void saveToConfig();

   private slots:
    void onUseHistoryToggled(bool enabled);
    void onCheckAllConversion();
    void onUncheckAllConversion();
    void onClearLearningData();

   private:
    Ui::MainWindow* ui_;
    QWidget* window_;
    TabContext context_;
    ServerConnector* server_;
};

}  // namespace hazkey::settings

#endif  // HAZKEY_SETTINGS_CONTROLLERS_CONVERSION_TAB_CONTROLLER_H_
