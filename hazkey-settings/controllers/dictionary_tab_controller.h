#ifndef HAZKEY_SETTINGS_CONTROLLERS_DICTIONARY_TAB_CONTROLLER_H_
#define HAZKEY_SETTINGS_CONTROLLERS_DICTIONARY_TAB_CONTROLLER_H_

#include <QObject>

#include "controllers/tab_context.h"

namespace hazkey::settings {

class DictionaryTabController : public QObject {
    Q_OBJECT

   public:
    explicit DictionaryTabController(QObject* parent);
    void setContext(const TabContext& context);
    void loadFromConfig();
    void saveToConfig();

   private:
    TabContext context_;
};

}  // namespace hazkey::settings

#endif  // HAZKEY_SETTINGS_CONTROLLERS_DICTIONARY_TAB_CONTROLLER_H_
