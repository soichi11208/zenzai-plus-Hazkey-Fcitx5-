#ifndef HAZKEY_SETTINGS_CONTROLLERS_INPUT_STYLE_TAB_CONTROLLER_H_
#define HAZKEY_SETTINGS_CONTROLLERS_INPUT_STYLE_TAB_CONTROLLER_H_

#include <QObject>

#include "controllers/dual_list_controller.h"
#include "controllers/tab_context.h"

namespace Ui {
class MainWindow;
}

namespace hazkey::settings {

class InputStyleTabController : public QObject {
    Q_OBJECT

   public:
    InputStyleTabController(Ui::MainWindow* ui, QObject* parent);
    void setContext(const TabContext& context);
    void connectSignals();
    void loadFromConfig();
    void saveToConfig();

   private slots:
    void onEnableTable();
    void onDisableTable();
    void onTableMoveUp();
    void onTableMoveDown();
    void onEnableKeymap();
    void onDisableKeymap();
    void onKeymapMoveUp();
    void onKeymapMoveDown();
    void onSubmodeEntryChanged();
    void onBasicInputStyleChanged();
    void onBasicSettingChanged();
    void onResetInputStyleToDefault();

   private:
    void loadInputTables();
    void saveInputTables();
    void loadKeymaps();
    void saveKeymaps();
    void syncBasicToAdvanced();
    void syncAdvancedToBasic();
    bool isBasicModeCompatible();
    void showBasicModeWarning();
    void hideBasicModeWarning();
    void setBasicTabEnabled(bool enabled);
    void applyBasicInputStyle();
    void applyBasicPunctuationStyle();
    void applyBasicNumberStyle();
    void applyBasicSymbolStyle();
    void applyBasicSpaceStyle();
    void addKeymapIfAvailable(const QString& keymapName, bool isBuiltIn);
    void addInputTableIfAvailable(const QString& tableName, bool isBuiltIn);
    void clearKeymapsAndTables();
    QString translateKeymapName(const QString& keymapName, bool isBuiltin);
    QString translateTableName(const QString& tableName, bool isBuiltin);
    void updateNoteTextPlaceholders();

    Ui::MainWindow* ui_;
    TabContext context_;
    DualListController tableListController_;
    DualListController keymapListController_;
    bool isUpdatingFromAdvanced_;
};

}  // namespace hazkey::settings

#endif  // HAZKEY_SETTINGS_CONTROLLERS_INPUT_STYLE_TAB_CONTROLLER_H_
