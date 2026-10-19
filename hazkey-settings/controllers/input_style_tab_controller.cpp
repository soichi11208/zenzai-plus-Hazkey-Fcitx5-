#include "controllers/input_style_tab_controller.h"

#include <QColor>
#include <QComboBox>
#include <QLayoutItem>
#include <QLineEdit>
#include <QListWidget>
#include <QPair>
#include <QSet>
#include <QString>
#include <QVBoxLayout>

#include "config_macros.h"
#include "controllers/warning_widget_factory.h"
#include "ui_mainwindow.h"

namespace hazkey::settings {

InputStyleTabController::InputStyleTabController(Ui::MainWindow* ui,
                                                 QObject* parent)
    : QObject(parent),
      ui_(ui),
      context_(),
      tableListController_(
          {ui_->enabledTableList, ui_->availableTableList, ui_->enableTable,
           ui_->disableTable, ui_->tableMoveUp, ui_->tableMoveDown},
          this),
      keymapListController_(
          {ui_->enabledKeymapList, ui_->availableKeymapList, ui_->enableKeymap,
           ui_->disableKeymap, ui_->keymapMoveUp, ui_->keymapMoveDown},
          this),
      isUpdatingFromAdvanced_(false) {}

void InputStyleTabController::setContext(const TabContext& context) {
    context_ = context;
}

void InputStyleTabController::connectSignals() {
    tableListController_.connectSignals();
    keymapListController_.connectSignals();

    connect(&tableListController_, &DualListController::enableRequested, this,
            &InputStyleTabController::onEnableTable);
    connect(&tableListController_, &DualListController::disableRequested, this,
            &InputStyleTabController::onDisableTable);
    connect(&tableListController_, &DualListController::moveUpRequested, this,
            &InputStyleTabController::onTableMoveUp);
    connect(&tableListController_, &DualListController::moveDownRequested, this,
            &InputStyleTabController::onTableMoveDown);

    connect(&keymapListController_, &DualListController::enableRequested, this,
            &InputStyleTabController::onEnableKeymap);
    connect(&keymapListController_, &DualListController::disableRequested, this,
            &InputStyleTabController::onDisableKeymap);
    connect(&keymapListController_, &DualListController::moveUpRequested, this,
            &InputStyleTabController::onKeymapMoveUp);
    connect(&keymapListController_, &DualListController::moveDownRequested,
            this, &InputStyleTabController::onKeymapMoveDown);

    connect(ui_->submodeEntryPointChars, &QLineEdit::textChanged, this,
            &InputStyleTabController::onSubmodeEntryChanged);
    connect(ui_->mainInputStyle,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &InputStyleTabController::onBasicInputStyleChanged);
    connect(ui_->punctuationStyle,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &InputStyleTabController::onBasicSettingChanged);
    connect(ui_->numberStyle,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &InputStyleTabController::onBasicSettingChanged);
    connect(ui_->commonSymbolStyle,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &InputStyleTabController::onBasicSettingChanged);
    connect(ui_->spaceStyleLabel,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &InputStyleTabController::onBasicSettingChanged);
}

void InputStyleTabController::loadFromConfig() {
    if (!context_.currentProfile) return;

    loadInputTables();
    loadKeymaps();

    ui_->submodeEntryPointChars->setText(QString::fromStdString(
        context_.currentProfile->submode_entry_point_chars()));

    syncAdvancedToBasic();
    updateNoteTextPlaceholders();
}

void InputStyleTabController::saveToConfig() {
    if (!context_.currentProfile) return;
    saveInputTables();
    saveKeymaps();
    context_.currentProfile->set_submode_entry_point_chars(
        GET_LINEEDIT_STRING(ui_->submodeEntryPointChars));
}

void InputStyleTabController::loadInputTables() {
    if (!context_.currentProfile) return;

    ui_->enabledTableList->clear();
    ui_->availableTableList->clear();

    QSet<QPair<QString, bool>> enabledTableKeys;

    for (int i = 0; i < context_.currentProfile->enabled_tables_size(); ++i) {
        const auto& enabledTable = context_.currentProfile->enabled_tables(i);
        QString tableName = QString::fromStdString(enabledTable.name());
        bool isBuiltIn = enabledTable.is_built_in();
        enabledTableKeys.insert(QPair<QString, bool>(tableName, isBuiltIn));

        QString displayName =
            translateTableName(tableName, enabledTable.is_built_in());
        QListWidgetItem* item = new QListWidgetItem(displayName);

        bool isAvailable = false;
        for (int j = 0; j < context_.currentConfig->available_tables_size();
             ++j) {
            const auto& availableTable =
                context_.currentConfig->available_tables(j);
            if (availableTable.name() == enabledTable.name() &&
                availableTable.is_built_in() == enabledTable.is_built_in()) {
                isAvailable = true;
                break;
            }
        }

        if (isBuiltIn) {
            displayName = displayName + " " + tr("[built-in]");
        }
        if (!isAvailable) {
            displayName = displayName + " " + tr("[not found]");
            item->setForeground(QColor(Qt::red));
        }
        item->setText(displayName);

        item->setData(Qt::UserRole, tableName);
        item->setData(Qt::UserRole + 1, isBuiltIn);
        item->setData(Qt::UserRole + 2, isAvailable);

        ui_->enabledTableList->addItem(item);
    }

    for (int i = 0; i < context_.currentConfig->available_tables_size(); ++i) {
        const auto& availableTable =
            context_.currentConfig->available_tables(i);
        QString tableName = QString::fromStdString(availableTable.name());
        bool isBuiltIn = availableTable.is_built_in();
        QPair<QString, bool> tableKey(tableName, isBuiltIn);

        if (!enabledTableKeys.contains(tableKey)) {
            QString displayName =
                translateTableName(tableName, availableTable.is_built_in());
            QListWidgetItem* item = new QListWidgetItem(displayName);

            if (availableTable.is_built_in()) {
                item->setText(displayName + " " + tr("[built-in]"));
            }

            item->setData(Qt::UserRole, tableName);
            item->setData(Qt::UserRole + 1, availableTable.is_built_in());
            item->setData(Qt::UserRole + 2, true);

            ui_->availableTableList->addItem(item);
        }
    }

    tableListController_.updateButtonStates();
}

void InputStyleTabController::saveInputTables() {
    if (!context_.currentProfile) return;

    context_.currentProfile->clear_enabled_tables();

    for (int i = 0; i < ui_->enabledTableList->count(); ++i) {
        QListWidgetItem* item = ui_->enabledTableList->item(i);
        QString tableName = item->data(Qt::UserRole).toString();
        bool isBuiltIn = item->data(Qt::UserRole + 1).toBool();
        bool isAvailable = item->data(Qt::UserRole + 2).toBool();

        auto* enabledTable = context_.currentProfile->add_enabled_tables();
        enabledTable->set_name(tableName.toStdString());
        enabledTable->set_is_built_in(isBuiltIn);

        if (isAvailable) {
            for (int j = 0; j < context_.currentConfig->available_tables_size();
                 ++j) {
                const auto& availableTable =
                    context_.currentConfig->available_tables(j);
                if (availableTable.name() == tableName.toStdString() &&
                    availableTable.is_built_in() == isBuiltIn) {
                    enabledTable->set_filename(availableTable.filename());
                    break;
                }
            }
        }
    }
}

void InputStyleTabController::loadKeymaps() {
    if (!context_.currentProfile) return;

    ui_->enabledKeymapList->clear();
    ui_->availableKeymapList->clear();

    QSet<QPair<QString, bool>> enabledKeymapKeys;

    for (int i = 0; i < context_.currentProfile->enabled_keymaps_size(); ++i) {
        const auto& enabledKeymap = context_.currentProfile->enabled_keymaps(i);
        QString keymapName = QString::fromStdString(enabledKeymap.name());
        bool isBuiltIn = enabledKeymap.is_built_in();
        enabledKeymapKeys.insert(QPair<QString, bool>(keymapName, isBuiltIn));

        QString displayName =
            translateKeymapName(keymapName, enabledKeymap.is_built_in());
        QListWidgetItem* item = new QListWidgetItem(displayName);

        bool isAvailable = false;
        for (int j = 0; j < context_.currentConfig->available_keymaps_size();
             ++j) {
            const auto& availableKeymap =
                context_.currentConfig->available_keymaps(j);
            if (availableKeymap.name() == enabledKeymap.name() &&
                availableKeymap.is_built_in() == enabledKeymap.is_built_in()) {
                isAvailable = true;
                isBuiltIn = availableKeymap.is_built_in();
                break;
            }
        }

        if (isBuiltIn) {
            displayName = displayName + " " + tr("[built-in]");
        }
        if (!isAvailable) {
            displayName = displayName + " " + tr("[not found]");
            item->setForeground(QColor(Qt::red));
        }
        item->setText(displayName);

        item->setData(Qt::UserRole, keymapName);
        item->setData(Qt::UserRole + 1, isBuiltIn);
        item->setData(Qt::UserRole + 2, isAvailable);

        ui_->enabledKeymapList->addItem(item);
    }

    for (int i = 0; i < context_.currentConfig->available_keymaps_size(); ++i) {
        const auto& availableKeymap =
            context_.currentConfig->available_keymaps(i);
        QString keymapName = QString::fromStdString(availableKeymap.name());
        bool isBuiltIn = availableKeymap.is_built_in();
        QPair<QString, bool> keymapKey(keymapName, isBuiltIn);

        if (!enabledKeymapKeys.contains(keymapKey)) {
            QString displayName =
                translateKeymapName(keymapName, availableKeymap.is_built_in());
            QListWidgetItem* item = new QListWidgetItem(displayName);

            if (availableKeymap.is_built_in()) {
                item->setText(displayName + " " + tr("[built-in]"));
            }

            item->setData(Qt::UserRole, keymapName);
            item->setData(Qt::UserRole + 1, availableKeymap.is_built_in());
            item->setData(Qt::UserRole + 2, true);

            ui_->availableKeymapList->addItem(item);
        }
    }

    keymapListController_.updateButtonStates();
}

void InputStyleTabController::saveKeymaps() {
    if (!context_.currentProfile) return;

    context_.currentProfile->clear_enabled_keymaps();

    for (int i = 0; i < ui_->enabledKeymapList->count(); ++i) {
        QListWidgetItem* item = ui_->enabledKeymapList->item(i);
        QString keymapName = item->data(Qt::UserRole).toString();
        bool isBuiltIn = item->data(Qt::UserRole + 1).toBool();
        bool isAvailable = item->data(Qt::UserRole + 2).toBool();

        auto* enabledKeymap = context_.currentProfile->add_enabled_keymaps();
        enabledKeymap->set_name(keymapName.toStdString());
        enabledKeymap->set_is_built_in(isBuiltIn);

        if (isAvailable) {
            for (int j = 0;
                 j < context_.currentConfig->available_keymaps_size(); ++j) {
                const auto& availableKeymap =
                    context_.currentConfig->available_keymaps(j);
                if (availableKeymap.name() == keymapName.toStdString() &&
                    availableKeymap.is_built_in() == isBuiltIn) {
                    enabledKeymap->set_filename(availableKeymap.filename());
                    break;
                }
            }
        }
    }
}

void InputStyleTabController::onEnableTable() {
    QListWidgetItem* item = ui_->availableTableList->currentItem();
    if (!item) return;

    const int row = ui_->availableTableList->row(item);
    ui_->availableTableList->takeItem(row);
    ui_->enabledTableList->addItem(item);

    tableListController_.updateButtonStates();
    saveInputTables();
    syncAdvancedToBasic();
}

void InputStyleTabController::onDisableTable() {
    QListWidgetItem* item = ui_->enabledTableList->currentItem();
    if (!item) return;

    const bool isAvailable = item->data(Qt::UserRole + 2).toBool();
    const int row = ui_->enabledTableList->row(item);
    ui_->enabledTableList->takeItem(row);

    if (isAvailable) {
        QString tableName = item->data(Qt::UserRole).toString();
        bool isBuiltIn = item->data(Qt::UserRole + 1).toBool();
        QString displayName = translateTableName(tableName, isBuiltIn);

        if (isBuiltIn) {
            item->setText(displayName + " " + tr("[built-in]"));
        } else {
            item->setText(displayName);
        }
        item->setForeground(QColor());
        ui_->availableTableList->addItem(item);
    } else {
        delete item;
    }

    tableListController_.updateButtonStates();
    saveInputTables();
    syncAdvancedToBasic();
}

void InputStyleTabController::onTableMoveUp() {
    QListWidgetItem* item = ui_->enabledTableList->currentItem();
    if (!item) return;

    const int row = ui_->enabledTableList->row(item);
    if (row > 0) {
        ui_->enabledTableList->takeItem(row);
        ui_->enabledTableList->insertItem(row - 1, item);
        ui_->enabledTableList->setCurrentItem(item);
    }

    tableListController_.updateButtonStates();
    saveInputTables();
    syncAdvancedToBasic();
}

void InputStyleTabController::onTableMoveDown() {
    QListWidgetItem* item = ui_->enabledTableList->currentItem();
    if (!item) return;

    const int row = ui_->enabledTableList->row(item);
    if (row < ui_->enabledTableList->count() - 1) {
        ui_->enabledTableList->takeItem(row);
        ui_->enabledTableList->insertItem(row + 1, item);
        ui_->enabledTableList->setCurrentItem(item);
    }

    tableListController_.updateButtonStates();
    saveInputTables();
    syncAdvancedToBasic();
}

void InputStyleTabController::onEnableKeymap() {
    QListWidgetItem* item = ui_->availableKeymapList->currentItem();
    if (!item) return;

    const int row = ui_->availableKeymapList->row(item);
    ui_->availableKeymapList->takeItem(row);
    ui_->enabledKeymapList->addItem(item);

    keymapListController_.updateButtonStates();
    saveKeymaps();
    syncAdvancedToBasic();
}

void InputStyleTabController::onDisableKeymap() {
    QListWidgetItem* item = ui_->enabledKeymapList->currentItem();
    if (!item) return;

    const bool isAvailable = item->data(Qt::UserRole + 2).toBool();
    const int row = ui_->enabledKeymapList->row(item);
    ui_->enabledKeymapList->takeItem(row);

    if (isAvailable) {
        QString keymapName = item->data(Qt::UserRole).toString();
        bool isBuiltIn = item->data(Qt::UserRole + 1).toBool();
        QString displayName = translateKeymapName(keymapName, isBuiltIn);

        if (isBuiltIn) {
            item->setText(displayName + " " + tr("[built-in]"));
        } else {
            item->setText(displayName);
        }
        item->setForeground(QColor());

        ui_->availableKeymapList->addItem(item);
    } else {
        delete item;
    }

    keymapListController_.updateButtonStates();
    saveKeymaps();
    syncAdvancedToBasic();
}

void InputStyleTabController::onKeymapMoveUp() {
    QListWidgetItem* item = ui_->enabledKeymapList->currentItem();
    if (!item) return;

    const int row = ui_->enabledKeymapList->row(item);
    if (row > 0) {
        ui_->enabledKeymapList->takeItem(row);
        ui_->enabledKeymapList->insertItem(row - 1, item);
        ui_->enabledKeymapList->setCurrentItem(item);
    }

    keymapListController_.updateButtonStates();
    saveKeymaps();
    syncAdvancedToBasic();
}

void InputStyleTabController::onKeymapMoveDown() {
    QListWidgetItem* item = ui_->enabledKeymapList->currentItem();
    if (!item) return;

    const int row = ui_->enabledKeymapList->row(item);
    if (row < ui_->enabledKeymapList->count() - 1) {
        ui_->enabledKeymapList->takeItem(row);
        ui_->enabledKeymapList->insertItem(row + 1, item);
        ui_->enabledKeymapList->setCurrentItem(item);
    }

    keymapListController_.updateButtonStates();
    saveKeymaps();
    syncAdvancedToBasic();
}

void InputStyleTabController::onSubmodeEntryChanged() {
    if (isUpdatingFromAdvanced_) return;
    if (context_.currentProfile) {
        context_.currentProfile->set_submode_entry_point_chars(
            ui_->submodeEntryPointChars->text().toStdString());
        syncAdvancedToBasic();
    }
}

void InputStyleTabController::onBasicInputStyleChanged() {
    if (isUpdatingFromAdvanced_) return;

    const bool isKana = (ui_->mainInputStyle->currentIndex() == 1);

    ui_->punctuationStyle->setEnabled(!isKana);
    ui_->numberStyle->setEnabled(!isKana);
    ui_->commonSymbolStyle->setEnabled(!isKana);

    if (isKana) {
        ui_->punctuationStyle->setToolTip(tr("Disabled in Kana mode"));
        ui_->numberStyle->setToolTip(tr("Disabled in Kana mode"));
        ui_->commonSymbolStyle->setToolTip(tr("Disabled in Kana mode"));
    } else {
        ui_->punctuationStyle->setToolTip("");
        ui_->numberStyle->setToolTip("");
        ui_->commonSymbolStyle->setToolTip("");
    }

    syncBasicToAdvanced();
}

void InputStyleTabController::onBasicSettingChanged() {
    if (isUpdatingFromAdvanced_) return;
    syncBasicToAdvanced();
}

void InputStyleTabController::onResetInputStyleToDefault() {
    ui_->mainInputStyle->setCurrentIndex(0);
    ui_->punctuationStyle->setCurrentIndex(0);
    ui_->numberStyle->setCurrentIndex(0);
    ui_->commonSymbolStyle->setCurrentIndex(0);
    ui_->spaceStyleLabel->setCurrentIndex(0);

    ui_->punctuationStyle->setEnabled(true);
    ui_->numberStyle->setEnabled(true);
    ui_->commonSymbolStyle->setEnabled(true);

    ui_->punctuationStyle->setToolTip("");
    ui_->numberStyle->setToolTip("");
    ui_->commonSymbolStyle->setToolTip("");

    syncBasicToAdvanced();
    hideBasicModeWarning();
}

void InputStyleTabController::syncBasicToAdvanced() {
    if (!context_.currentProfile) return;

    clearKeymapsAndTables();

    applyBasicPunctuationStyle();
    applyBasicNumberStyle();
    applyBasicSymbolStyle();
    applyBasicSpaceStyle();
    applyBasicInputStyle();

    if (context_.currentProfile) {
        isUpdatingFromAdvanced_ = true;
        ui_->submodeEntryPointChars->setText(QString::fromStdString(
            context_.currentProfile->submode_entry_point_chars()));
        isUpdatingFromAdvanced_ = false;
    }

    loadInputTables();
    loadKeymaps();
}

void InputStyleTabController::syncAdvancedToBasic() {
    if (!context_.currentProfile) return;

    isUpdatingFromAdvanced_ = true;

    if (isBasicModeCompatible()) {
        hideBasicModeWarning();
        setBasicTabEnabled(true);

        QString submodeEntry = QString::fromStdString(
            context_.currentProfile->submode_entry_point_chars());
        bool hasRomajiTable = false;
        bool hasKanaTable = false;

        for (int i = 0; i < context_.currentProfile->enabled_tables_size();
             ++i) {
            const auto& table = context_.currentProfile->enabled_tables(i);
            QString tableName = QString::fromStdString(table.name());
            if (tableName.contains("Romaji", Qt::CaseInsensitive)) {
                hasRomajiTable = true;
            }
            if (tableName.contains("Kana", Qt::CaseInsensitive)) {
                hasKanaTable = true;
            }
        }

        bool isKanaMode = false;
        if (hasRomajiTable) {
            ui_->mainInputStyle->setCurrentIndex(0);
        } else if (hasKanaTable) {
            ui_->mainInputStyle->setCurrentIndex(1);
            isKanaMode = true;
        }

        ui_->punctuationStyle->setEnabled(!isKanaMode);
        ui_->numberStyle->setEnabled(!isKanaMode);
        ui_->commonSymbolStyle->setEnabled(!isKanaMode);

        if (isKanaMode) {
            ui_->punctuationStyle->setToolTip(tr("Disabled in Kana mode"));
            ui_->numberStyle->setToolTip(tr("Disabled in Kana mode"));
            ui_->commonSymbolStyle->setToolTip(tr("Disabled in Kana mode"));
        } else {
            ui_->punctuationStyle->setToolTip("");
            ui_->numberStyle->setToolTip("");
            ui_->commonSymbolStyle->setToolTip("");
        }

        QSet<QString> enabledKeymaps;
        for (int i = 0; i < context_.currentProfile->enabled_keymaps_size();
             ++i) {
            const auto& keymap = context_.currentProfile->enabled_keymaps(i);
            enabledKeymaps.insert(QString::fromStdString(keymap.name()));
        }

        if (enabledKeymaps.contains("Fullwidth Period") &&
            enabledKeymaps.contains("Fullwidth Comma")) {
            ui_->punctuationStyle->setCurrentIndex(1);
        } else if (enabledKeymaps.contains("Fullwidth Comma") &&
                   !enabledKeymaps.contains("Fullwidth Period")) {
            ui_->punctuationStyle->setCurrentIndex(2);
        } else if (enabledKeymaps.contains("Fullwidth Period") &&
                   !enabledKeymaps.contains("Fullwidth Comma")) {
            ui_->punctuationStyle->setCurrentIndex(3);
        } else {
            ui_->punctuationStyle->setCurrentIndex(0);
        }

        if (enabledKeymaps.contains("Fullwidth Number")) {
            ui_->numberStyle->setCurrentIndex(0);
        } else {
            ui_->numberStyle->setCurrentIndex(1);
        }

        if (enabledKeymaps.contains("Fullwidth Symbol")) {
            ui_->commonSymbolStyle->setCurrentIndex(0);
        } else {
            ui_->commonSymbolStyle->setCurrentIndex(1);
        }

        if (enabledKeymaps.contains("Fullwidth Space")) {
            ui_->spaceStyleLabel->setCurrentIndex(0);
        } else {
            ui_->spaceStyleLabel->setCurrentIndex(1);
        }

    } else {
        showBasicModeWarning();
        setBasicTabEnabled(false);
        ui_->inputTableConfigModeTabWidget->setCurrentIndex(1);
    }

    isUpdatingFromAdvanced_ = false;
}

bool InputStyleTabController::isBasicModeCompatible() {
    if (!context_.currentProfile) return false;

    QString submodeEntry = QString::fromStdString(
        context_.currentProfile->submode_entry_point_chars());

    QList<QString> enabledCustomKeymaps;
    QList<QString> enabledCustomTables;
    QList<QString> enabledBuiltinKeymaps;
    QList<QString> enabledBuiltinTables;

    for (int i = 0; i < context_.currentProfile->enabled_keymaps_size(); ++i) {
        const auto& keymap = context_.currentProfile->enabled_keymaps(i);
        QString name = QString::fromStdString(keymap.name());
        if (keymap.is_built_in()) {
            enabledBuiltinKeymaps.append(name);
        } else {
            enabledCustomKeymaps.append(name);
        }
    }

    for (int i = 0; i < context_.currentProfile->enabled_tables_size(); ++i) {
        const auto& table = context_.currentProfile->enabled_tables(i);
        QString name = QString::fromStdString(table.name());
        if (table.is_built_in()) {
            enabledBuiltinTables.append(name);
        } else {
            enabledCustomTables.append(name);
        }
    }

    if (enabledCustomTables.size() != 0 || enabledCustomKeymaps.size() != 0) {
        return false;
    }

    bool hasBuiltinRomajiTable = enabledBuiltinTables.contains("Romaji");
    bool hasBuiltinKanaTable = enabledBuiltinTables.contains("Kana");
    bool hasBuiltinKanaKeymap = enabledBuiltinKeymaps.contains("JIS Kana");
    bool isRomajiSubmode = (submodeEntry == "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    bool isKanaSubmode = submodeEntry.isEmpty();

    bool isValidRomajiInputStyle =
        (hasBuiltinRomajiTable && !hasBuiltinKanaTable && isRomajiSubmode &&
         !hasBuiltinKanaKeymap);
    bool isValidKanaInputStyle =
        (hasBuiltinKanaTable && !hasBuiltinRomajiTable && isKanaSubmode &&
         hasBuiltinKanaKeymap);

    if (isValidKanaInputStyle) {
        QSet<QString> allowedKanaModeKeymaps = {"Fullwidth Space", "JIS Kana"};
        for (const QString& keymap : enabledBuiltinKeymaps) {
            if (!allowedKanaModeKeymaps.contains(keymap)) {
                return false;
            }
        }
        return true;
    } else if (isValidRomajiInputStyle) {
        QSet<QString> validBasicKeymaps = {
            "Fullwidth Period", "Fullwidth Comma", "Fullwidth Number",
            "Fullwidth Symbol", "Fullwidth Space", "Japanese Symbol"};

        bool checkedJapaneseSymbolMap = false;

        for (const QString& keymap : enabledBuiltinKeymaps) {
            if (!validBasicKeymaps.contains(keymap)) {
                return false;
            }
            if (checkedJapaneseSymbolMap &&
                (keymap == "Fullwidth Period" || keymap == "Fullwidth Comma")) {
                return false;
            }
            if (keymap == "Japanese Symbol") {
                checkedJapaneseSymbolMap = true;
            }
        }
        if (!checkedJapaneseSymbolMap) {
            return false;
        }
        return true;
    }
    return false;
}

void InputStyleTabController::showBasicModeWarning() {
    hideBasicModeWarning();

    QVBoxLayout* basicTabLayout = qobject_cast<QVBoxLayout*>(
        ui_->inputStyleSimpleModeScrollAreaContents->layout());

    if (basicTabLayout) {
        QWidget* warningWidget = WarningWidgetFactory::create(
            tr("<b>Warning:</b> Current settings can only be edited in "
               "Advanced "
               "mode."),
            "yellow", tr("Reset Input Style"),
            [this]() { onResetInputStyleToDefault(); });

        basicTabLayout->insertWidget(0, warningWidget);
        setBasicTabEnabled(false);
    }
}

void InputStyleTabController::hideBasicModeWarning() {
    QVBoxLayout* basicTabLayout = qobject_cast<QVBoxLayout*>(
        ui_->inputStyleSimpleModeScrollAreaContents->layout());
    if (basicTabLayout) {
        for (int i = basicTabLayout->count() - 1; i >= 0; --i) {
            QLayoutItem* item = basicTabLayout->itemAt(i);
            if (item && item->widget()) {
                QWidget* widget = item->widget();
                if (widget->styleSheet().contains("background-color: yellow")) {
                    basicTabLayout->removeWidget(widget);
                    widget->deleteLater();
                    setBasicTabEnabled(true);

                    if (ui_->mainInputStyle->currentIndex() == 1) {
                        ui_->punctuationStyle->setEnabled(false);
                        ui_->numberStyle->setEnabled(false);
                        ui_->commonSymbolStyle->setEnabled(false);
                        ui_->punctuationStyle->setToolTip(
                            "Disabled in Kana mode");
                        ui_->numberStyle->setToolTip("Disabled in Kana mode");
                        ui_->commonSymbolStyle->setToolTip(
                            "Disabled in Kana mode");
                    }
                    return;
                }
            }
        }
    }
}

void InputStyleTabController::setBasicTabEnabled(bool enabled) {
    ui_->inputStylesGrid->setEnabled(enabled);
    ui_->mainInputStyle->setEnabled(enabled);
    ui_->punctuationStyle->setEnabled(enabled);
    ui_->numberStyle->setEnabled(enabled);
    ui_->commonSymbolStyle->setEnabled(enabled);
    ui_->spaceStyleLabel->setEnabled(enabled);
}

void InputStyleTabController::applyBasicInputStyle() {
    const int inputStyleIndex = ui_->mainInputStyle->currentIndex();

    if (inputStyleIndex == 0) {
        context_.currentProfile->set_submode_entry_point_chars(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
        addInputTableIfAvailable("Romaji", true);
        addKeymapIfAvailable("Japanese Symbol", true);
    } else if (inputStyleIndex == 1) {
        context_.currentProfile->set_submode_entry_point_chars("");
        addInputTableIfAvailable("Kana", true);
        addKeymapIfAvailable("JIS Kana", true);
    }
}

void InputStyleTabController::applyBasicPunctuationStyle() {
    if (!ui_->punctuationStyle->isEnabled()) return;

    const int punctuationIndex = ui_->punctuationStyle->currentIndex();

    switch (punctuationIndex) {
        case 1:
            addKeymapIfAvailable("Fullwidth Period", true);
            addKeymapIfAvailable("Fullwidth Comma", true);
            break;
        case 2:
            addKeymapIfAvailable("Fullwidth Comma", true);
            break;
        case 3:
            addKeymapIfAvailable("Fullwidth Period", true);
            break;
        default:
            break;
    }
}

void InputStyleTabController::applyBasicNumberStyle() {
    if (!ui_->numberStyle->isEnabled()) return;
    const int numberIndex = ui_->numberStyle->currentIndex();
    if (numberIndex == 0) {
        addKeymapIfAvailable("Fullwidth Number", true);
    }
}

void InputStyleTabController::applyBasicSymbolStyle() {
    if (!ui_->commonSymbolStyle->isEnabled()) return;
    const int symbolIndex = ui_->commonSymbolStyle->currentIndex();
    if (symbolIndex == 0) {
        addKeymapIfAvailable("Fullwidth Symbol", true);
    }
}

void InputStyleTabController::applyBasicSpaceStyle() {
    const int spaceIndex = ui_->spaceStyleLabel->currentIndex();
    if (spaceIndex == 0) {
        addKeymapIfAvailable("Fullwidth Space", true);
    }
}

void InputStyleTabController::addKeymapIfAvailable(const QString& keymapName,
                                                   bool isBuiltIn) {
    for (int i = 0; i < context_.currentConfig->available_keymaps_size(); ++i) {
        const auto& availableKeymap =
            context_.currentConfig->available_keymaps(i);
        if (QString::fromStdString(availableKeymap.name()) == keymapName &&
            availableKeymap.is_built_in() == isBuiltIn) {
            auto* enabledKeymap =
                context_.currentProfile->add_enabled_keymaps();
            enabledKeymap->set_name(availableKeymap.name());
            enabledKeymap->set_is_built_in(availableKeymap.is_built_in());
            enabledKeymap->set_filename(availableKeymap.filename());
            break;
        }
    }
}

void InputStyleTabController::addInputTableIfAvailable(const QString& tableName,
                                                       bool isBuiltIn) {
    for (int i = 0; i < context_.currentConfig->available_tables_size(); ++i) {
        const auto& availableTable =
            context_.currentConfig->available_tables(i);
        if (QString::fromStdString(availableTable.name()) == tableName &&
            availableTable.is_built_in() == isBuiltIn) {
            auto* enabledTable = context_.currentProfile->add_enabled_tables();
            enabledTable->set_name(availableTable.name());
            enabledTable->set_is_built_in(availableTable.is_built_in());
            enabledTable->set_filename(availableTable.filename());
            break;
        }
    }
}

void InputStyleTabController::clearKeymapsAndTables() {
    if (context_.currentProfile) {
        context_.currentProfile->clear_enabled_keymaps();
        context_.currentProfile->clear_enabled_tables();
    }
}

QString InputStyleTabController::translateKeymapName(const QString& keymapName,
                                                     bool isBuiltin) {
    if (!isBuiltin) return keymapName;
    if (keymapName == "JIS Kana") {
        return tr("JIS Kana");
    } else if (keymapName == "Japanese Symbol") {
        return tr("Japanese Symbol");
    } else if (keymapName == "Fullwidth Period") {
        return tr("Fullwidth Period");
    } else if (keymapName == "Fullwidth Comma") {
        return tr("Fullwidth Comma");
    } else if (keymapName == "Fullwidth Number") {
        return tr("Fullwidth Number");
    } else if (keymapName == "Fullwidth Symbol") {
        return tr("Fullwidth Symbol");
    } else if (keymapName == "Fullwidth Space") {
        return tr("Fullwidth Space");
    }
    return keymapName;
}

QString InputStyleTabController::translateTableName(const QString& tableName,
                                                    bool isBuiltin) {
    if (!isBuiltin) return tableName;
    if (tableName == "Romaji") {
        return tr("Romaji");
    } else if (tableName == "Kana") {
        return tr("Kana");
    }
    return tableName;
}

void InputStyleTabController::updateNoteTextPlaceholders() {
    if (!context_.currentConfig) return;
    QString xdgConfigHome =
        QString::fromStdString(context_.currentConfig->xdg_config_home_path());
    if (!xdgConfigHome.isEmpty()) {
        if (xdgConfigHome.endsWith('/')) {
            xdgConfigHome.chop(1);
        }

        QString keymapNoteText = ui_->keymapAdvancedNote->text();
        keymapNoteText.replace("$XDG_CONFIG_HOME/hazkey", xdgConfigHome);
        ui_->keymapAdvancedNote->setText(keymapNoteText);

        QString inputTableNoteText = ui_->inputTableAdvancedNote->text();
        inputTableNoteText.replace("$XDG_CONFIG_HOME/hazkey", xdgConfigHome);
        ui_->inputTableAdvancedNote->setText(inputTableNoteText);
    }
}

}  // namespace hazkey::settings
