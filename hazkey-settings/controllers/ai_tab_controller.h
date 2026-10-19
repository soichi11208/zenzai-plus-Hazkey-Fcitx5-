#ifndef HAZKEY_SETTINGS_CONTROLLERS_AI_TAB_CONTROLLER_H_
#define HAZKEY_SETTINGS_CONTROLLERS_AI_TAB_CONTROLLER_H_

#include <QNetworkReply>
#include <QObject>
#include <QString>
#include <QtGlobal>
#include <atomic>

#include "controllers/tab_context.h"

class QWidget;
class QNetworkAccessManager;
class QProgressDialog;

namespace Ui {
class MainWindow;
}

namespace hazkey::settings {

class AiTabController : public QObject {
    Q_OBJECT

   public:
    AiTabController(Ui::MainWindow* ui, QWidget* window,
                    QNetworkAccessManager* networkManager, QObject* parent);
    void setContext(const TabContext& context);
    void connectSignals();
    void loadFromConfig();
    void saveToConfig();

   private slots:
    void onDownloadZenzaiModel();
    void onDownloadProgress(qint64 bytesReceived, qint64 bytesTotal);
    void onDownloadFinished();
    void onDownloadError(QNetworkReply::NetworkError error);

   private:
    QString calculateFileSHA256(const QString& filePath);
    void refreshWarnings();
    void populateDeviceList();
    void updateSelectionFromProfile();

    Ui::MainWindow* ui_;
    QWidget* window_;
    TabContext context_;
    QNetworkAccessManager* networkManager_;
    QNetworkReply* currentDownload_;
    QProgressDialog* downloadProgressDialog_;
    QString zenzaiModelPath_;
    std::atomic<bool> isLoading_{false};
};

}  // namespace hazkey::settings

#endif  // HAZKEY_SETTINGS_CONTROLLERS_AI_TAB_CONTROLLER_H_
