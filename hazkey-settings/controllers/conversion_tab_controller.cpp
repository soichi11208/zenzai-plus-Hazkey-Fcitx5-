#include "controllers/conversion_tab_controller.h"

#include <QCheckBox>
#include <QMessageBox>
#include <QPushButton>

#include "config_macros.h"
#include "ui_mainwindow.h"

namespace hazkey::settings {

ConversionTabController::ConversionTabController(Ui::MainWindow* ui,
                                                 QWidget* window,
                                                 ServerConnector* server,
                                                 QObject* parent)
    : QObject(parent), ui_(ui), window_(window), server_(server) {}

void ConversionTabController::setContext(const TabContext& context) {
    context_ = context;
}

void ConversionTabController::connectSignals() {
    connect(ui_->useHistory, &QCheckBox::toggled, this,
            &ConversionTabController::onUseHistoryToggled);
    connect(ui_->checkAllConversion, &QPushButton::clicked, this,
            &ConversionTabController::onCheckAllConversion);
    connect(ui_->uncheckAllConversion, &QPushButton::clicked, this,
            &ConversionTabController::onUncheckAllConversion);
    connect(ui_->clearLearningData, &QPushButton::clicked, this,
            &ConversionTabController::onClearLearningData);
}

void ConversionTabController::loadFromConfig() {
    if (!context_.currentProfile) return;

    SET_CHECKBOX(ui_->useHistory, context_.currentProfile->use_input_history(),
                 ConfigDefs::CheckboxDefaults::USE_HISTORY);
    SET_CHECKBOX(ui_->stopStoreNewHistory,
                 context_.currentProfile->stop_store_new_history(),
                 ConfigDefs::CheckboxDefaults::STOP_STORE_NEW_HISTORY);

    auto specialConversions =
        context_.currentProfile->mutable_special_conversion_mode();
    SET_CHECKBOX(ui_->halfwidthKatakanaConversion,
                 specialConversions->halfwidth_katakana(),
                 ConfigDefs::CheckboxDefaults::HALFWIDTH_KATAKANA);
    SET_CHECKBOX(ui_->extendedEmojiConversion,
                 specialConversions->extended_emoji(),
                 ConfigDefs::CheckboxDefaults::EXTENDED_EMOJI);
    SET_CHECKBOX(ui_->commaSeparatedNumCoversion,
                 specialConversions->comma_separated_number(),
                 ConfigDefs::CheckboxDefaults::COMMA_SEPARATED_NUMBER);
    SET_CHECKBOX(ui_->calendarConversion, specialConversions->calendar(),
                 ConfigDefs::CheckboxDefaults::CALENDER);
    SET_CHECKBOX(ui_->timeConversion, specialConversions->time(),
                 ConfigDefs::CheckboxDefaults::TIME);
    SET_CHECKBOX(ui_->mailDomainConversion, specialConversions->mail_domain(),
                 ConfigDefs::CheckboxDefaults::MAIL_DOMAIN);
    SET_CHECKBOX(ui_->unicodeCodePointConversion,
                 specialConversions->unicode_codepoint(),
                 ConfigDefs::CheckboxDefaults::UNICODE_CODEPOINT);
    SET_CHECKBOX(ui_->romanTypographyConversion,
                 specialConversions->roman_typography(),
                 ConfigDefs::CheckboxDefaults::ROMAN_TYPOGRAPHY);
    SET_CHECKBOX(ui_->hazkeyVersionConversion,
                 specialConversions->hazkey_version(),
                 ConfigDefs::CheckboxDefaults::HAZKEY_VERSION);

    ui_->stopStoreNewHistory->setEnabled(
        context_.currentProfile->use_input_history());
}

void ConversionTabController::saveToConfig() {
    if (!context_.currentProfile) return;

    context_.currentProfile->set_use_input_history(
        GET_CHECKBOX_BOOL(ui_->useHistory));
    context_.currentProfile->set_stop_store_new_history(
        GET_CHECKBOX_BOOL(ui_->stopStoreNewHistory));

    auto* specialConversions =
        context_.currentProfile->mutable_special_conversion_mode();
    specialConversions->set_halfwidth_katakana(
        GET_CHECKBOX_BOOL(ui_->halfwidthKatakanaConversion));
    specialConversions->set_extended_emoji(
        GET_CHECKBOX_BOOL(ui_->extendedEmojiConversion));
    specialConversions->set_comma_separated_number(
        GET_CHECKBOX_BOOL(ui_->commaSeparatedNumCoversion));
    specialConversions->set_calendar(
        GET_CHECKBOX_BOOL(ui_->calendarConversion));
    specialConversions->set_time(GET_CHECKBOX_BOOL(ui_->timeConversion));
    specialConversions->set_mail_domain(
        GET_CHECKBOX_BOOL(ui_->mailDomainConversion));
    specialConversions->set_unicode_codepoint(
        GET_CHECKBOX_BOOL(ui_->unicodeCodePointConversion));
    specialConversions->set_roman_typography(
        GET_CHECKBOX_BOOL(ui_->romanTypographyConversion));
    specialConversions->set_hazkey_version(
        GET_CHECKBOX_BOOL(ui_->hazkeyVersionConversion));
}

void ConversionTabController::onUseHistoryToggled(bool enabled) {
    ui_->stopStoreNewHistory->setEnabled(enabled);
}

void ConversionTabController::onCheckAllConversion() {
    ui_->halfwidthKatakanaConversion->setChecked(true);
    ui_->extendedEmojiConversion->setChecked(true);
    ui_->commaSeparatedNumCoversion->setChecked(true);
    ui_->calendarConversion->setChecked(true);
    ui_->timeConversion->setChecked(true);
    ui_->mailDomainConversion->setChecked(true);
    ui_->unicodeCodePointConversion->setChecked(true);
    ui_->romanTypographyConversion->setChecked(true);
    ui_->hazkeyVersionConversion->setChecked(true);
}

void ConversionTabController::onUncheckAllConversion() {
    ui_->halfwidthKatakanaConversion->setChecked(false);
    ui_->extendedEmojiConversion->setChecked(false);
    ui_->commaSeparatedNumCoversion->setChecked(false);
    ui_->calendarConversion->setChecked(false);
    ui_->timeConversion->setChecked(false);
    ui_->mailDomainConversion->setChecked(false);
    ui_->unicodeCodePointConversion->setChecked(false);
    ui_->romanTypographyConversion->setChecked(false);
    ui_->hazkeyVersionConversion->setChecked(false);
}

void ConversionTabController::onClearLearningData() {
    if (!context_.currentProfile) {
        QMessageBox::warning(window_, QObject::tr("Error"),
                             QObject::tr("No configuration profile loaded."));
        return;
    }

    QMessageBox::StandardButton reply = QMessageBox::question(
        window_, QObject::tr("Clear Input History"),
        QObject::tr("Are you sure you want to clear all input history data? "
                    "This action cannot be undone."),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);

    if (reply == QMessageBox::Yes && server_) {
        const bool success =
            server_->clearAllHistory(context_.currentProfile->profile_id());

        if (success) {
            QMessageBox::information(
                window_, QObject::tr("Success"),
                QObject::tr("Input history has been cleared successfully."));
        } else {
            QMessageBox::critical(window_, QObject::tr("Error"),
                                  QObject::tr("Failed to clear input history. "
                                              "Please check your connection "
                                              "to the hazkey server."));
        }
    }
}

}  // namespace hazkey::settings
