#include "controllers/dual_list_controller.h"

DualListController::DualListController(const Widgets& widgets, QObject* parent)
    : QObject(parent), widgets_(widgets) {}

void DualListController::connectSignals() {
    connect(widgets_.enabledList, &QListWidget::itemSelectionChanged, this,
            &DualListController::handleEnabledSelectionChanged);
    connect(widgets_.availableList, &QListWidget::itemSelectionChanged, this,
            &DualListController::handleAvailableSelectionChanged);

    connect(widgets_.moveUpButton, &QToolButton::clicked, this,
            &DualListController::handleMoveUp);
    connect(widgets_.moveDownButton, &QToolButton::clicked, this,
            &DualListController::handleMoveDown);

    connect(widgets_.enableButton, &QToolButton::clicked, this,
            &DualListController::handleEnable);
    connect(widgets_.disableButton, &QToolButton::clicked, this,
            &DualListController::handleDisable);

    connect(widgets_.enabledList, &QListWidget::itemDoubleClicked, this,
            &DualListController::handleDisable);
    connect(widgets_.availableList, &QListWidget::itemDoubleClicked, this,
            &DualListController::handleEnable);

    updateButtonStates();
}

void DualListController::handleEnabledSelectionChanged() {
    updateButtonStates();
}

void DualListController::handleAvailableSelectionChanged() {
    updateButtonStates();
}

void DualListController::handleMoveUp() { emit moveUpRequested(); }

void DualListController::handleMoveDown() { emit moveDownRequested(); }

void DualListController::handleEnable() { emit enableRequested(); }

void DualListController::handleDisable() { emit disableRequested(); }

void DualListController::updateButtonStates() {
    QListWidgetItem* enabledItem = widgets_.enabledList->currentItem();
    QListWidgetItem* availableItem = widgets_.availableList->currentItem();

    widgets_.disableButton->setEnabled(enabledItem != nullptr);
    widgets_.enableButton->setEnabled(availableItem != nullptr);

    if (enabledItem) {
        const int row = widgets_.enabledList->row(enabledItem);
        widgets_.moveUpButton->setEnabled(row > 0);
        widgets_.moveDownButton->setEnabled(row <
                                            widgets_.enabledList->count() - 1);
    } else {
        widgets_.moveUpButton->setEnabled(false);
        widgets_.moveDownButton->setEnabled(false);
    }
}
