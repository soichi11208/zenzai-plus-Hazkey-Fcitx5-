#ifndef HAZKEY_SETTINGS_CONTROLLERS_USER_INTERFACE_TAB_CONTROLLER_H_
#define HAZKEY_SETTINGS_CONTROLLERS_USER_INTERFACE_TAB_CONTROLLER_H_

#include <QObject>

#include "controllers/tab_context.h"

namespace Ui {
class MainWindow;
}

namespace hazkey::settings {

class UserInterfaceTabController : public QObject {
    Q_OBJECT

   public:
    explicit UserInterfaceTabController(Ui::MainWindow* ui, QObject* parent);
    void setContext(const TabContext& context);
    void loadFromConfig();
    void saveToConfig();

   private:
    Ui::MainWindow* ui_;
    TabContext context_;
};

}  // namespace hazkey::settings

#endif  // HAZKEY_SETTINGS_CONTROLLERS_USER_INTERFACE_TAB_CONTROLLER_H_
