#include "controllers/about_tab_controller.h"

#include <QString>

#include "constants.h"
#include "ui_mainwindow.h"

namespace hazkey::settings {

AboutTabController::AboutTabController(Ui::MainWindow* ui, QObject* parent)
    : QObject(parent), ui_(ui) {}

void AboutTabController::initialize() {
    QString hazkeyVersionText =
        QString(
            "<html><head/><body><p><span style=\"font-size:18pt\">%1"
            "</span></p></body></html>")
            .arg(HAZKEY_VERSION_STR);
    ui_->aboutHazkeyTitleVersionText->setText(hazkeyVersionText);
}

}  // namespace hazkey::settings
