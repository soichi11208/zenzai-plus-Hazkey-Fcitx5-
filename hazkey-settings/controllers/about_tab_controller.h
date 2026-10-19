#ifndef HAZKEY_SETTINGS_CONTROLLERS_ABOUT_TAB_CONTROLLER_H_
#define HAZKEY_SETTINGS_CONTROLLERS_ABOUT_TAB_CONTROLLER_H_

#include <QObject>

namespace Ui {
class MainWindow;
}

namespace hazkey::settings {

class AboutTabController : public QObject {
    Q_OBJECT

   public:
    AboutTabController(Ui::MainWindow* ui, QObject* parent);
    void initialize();

   private:
    Ui::MainWindow* ui_;
};

}  // namespace hazkey::settings

#endif  // HAZKEY_SETTINGS_CONTROLLERS_ABOUT_TAB_CONTROLLER_H_
