#ifndef HAZKEY_SETTINGS_CONTROLLERS_DUAL_LIST_CONTROLLER_H_
#define HAZKEY_SETTINGS_CONTROLLERS_DUAL_LIST_CONTROLLER_H_

#include <QListWidget>
#include <QObject>
#include <QToolButton>

class DualListController : public QObject {
    Q_OBJECT

   public:
    struct Widgets {
        QListWidget* enabledList;
        QListWidget* availableList;
        QToolButton* enableButton;
        QToolButton* disableButton;
        QToolButton* moveUpButton;
        QToolButton* moveDownButton;
    };

    explicit DualListController(const Widgets& widgets,
                                QObject* parent = nullptr);

    void connectSignals();
    void updateButtonStates();

   signals:
    void enableRequested();
    void disableRequested();
    void moveUpRequested();
    void moveDownRequested();

   private slots:
    void handleEnabledSelectionChanged();
    void handleAvailableSelectionChanged();
    void handleMoveUp();
    void handleMoveDown();
    void handleEnable();
    void handleDisable();

   private:
    Widgets widgets_;
};

#endif  // HAZKEY_SETTINGS_CONTROLLERS_DUAL_LIST_CONTROLLER_H_