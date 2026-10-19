#include "controllers/dictionary_tab_controller.h"

#include <QDir>
#include <QFile>
#include <QFormLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSaveFile>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTimer>
#include <QUuid>
#include <QVBoxLayout>

namespace hazkey::settings {

DictionaryTabController::DictionaryTabController(QObject* parent)
    : QObject(parent) {
    auto* layout = parent->findChild<QVBoxLayout*>("dictionaryTabScrollContentsLayout");
    if (!layout) return;
    // Replace the previously disabled user-dictionary placeholder.
    const auto hideItems = [](auto&& self, QLayout* container) -> void {
        for (int i = 0; i < container->count(); ++i) {
            auto* item = container->itemAt(i);
            if (item->widget()) item->widget()->hide();
            if (item->layout()) self(self, item->layout());
        }
    };
    hideItems(hideItems, layout);
    auto* panel = new QWidget;
    auto* form = new QFormLayout(panel);
    auto* note = new QLabel(QStringLiteral("読みと単語を辞書に登録し、モデルにも学習させます。辞書登録は学習完了を待たずに反映されます。"));
    note->setWordWrap(true);
    form->addRow(note);
    auto* reading = new QLineEdit;
    auto* word = new QLineEdit;
    reading->setMaxLength(128);
    word->setMaxLength(128);
    reading->setPlaceholderText(QStringLiteral("しょくもつせんい"));
    word->setPlaceholderText(QStringLiteral("食物繊維"));
    auto* steps = new QSpinBox;
    steps->setRange(1, 128);
    steps->setValue(32);
    form->addRow(QStringLiteral("読み"), reading);
    form->addRow(QStringLiteral("単語"), word);
    form->addRow(QStringLiteral("学習回数の上限"), steps);
    auto* submit = new QPushButton(QStringLiteral("TTTで登録"));
    auto* status = new QLabel;
    status->setWordWrap(true);
    status->setTextInteractionFlags(Qt::TextSelectableByMouse);
    form->addRow(submit);
    form->addRow(status);
    layout->insertWidget(0, panel);
    auto* timer = new QTimer(this);
    timer->setInterval(500);
    auto statusPath = std::make_shared<QString>();
    connect(submit, &QPushButton::clicked, this, [=]() {
        const auto kana = reading->text().trimmed();
        const auto text = word->text().trimmed();
        if (kana.isEmpty() || text.isEmpty()) {
            status->setText(QStringLiteral("読みと単語を入力してください。"));
            return;
        }
        for (auto c : kana) {
            if (!((c.unicode() >= 0x3041 && c.unicode() <= 0x3096) ||
                  (c.unicode() >= 0x30a1 && c.unicode() <= 0x30fa) || c == QChar(0x30fc))) {
                status->setText(QStringLiteral("読みはひらがな・カタカナで入力してください。"));
                return;
            }
        }
        const auto root = QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation) + "/hazkey/zenzai-ttt";
        if (!QFile::exists(root + "/base-model.json") && !QFile::exists(root + "/source-model/config.json")) {
            status->setText(QStringLiteral("Hazkeyの学習モデルがありません。Hazkeyを起動してください。"));
            return;
        }
        const auto directory = root + "/word-requests";
        if (!QDir().mkpath(directory)) {
            status->setText(QStringLiteral("登録先を作成できませんでした。"));
            return;
        }
        QFile::setPermissions(directory, QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner);
        const auto id = QUuid::createUuid().toString(QUuid::WithoutBraces);
        const auto path = directory + "/" + id;
        QSaveFile file(path + ".json");
        const auto payload = QJsonDocument(QJsonObject{{"reading", kana}, {"output", text}, {"max_steps", steps->value()}}).toJson();
        if (!file.open(QIODevice::WriteOnly)) {
            status->setText(QStringLiteral("登録要求を書き込めませんでした。"));
            return;
        }
        file.setPermissions(QFile::ReadOwner | QFile::WriteOwner);
        if (file.write(payload) != payload.size() || !file.commit()) {
            status->setText(QStringLiteral("登録要求の保存に失敗しました。"));
            return;
        }
        *statusPath = path + ".status";
        status->setText(QStringLiteral("学習待ちです。Hazkeyが起動していれば順に処理されます。"));
        submit->setEnabled(false);
        timer->start();
    });
    connect(timer, &QTimer::timeout, this, [=]() {
        QFile file(*statusPath);
        if (!file.open(QIODevice::ReadOnly)) return;
        const auto result = QJsonDocument::fromJson(file.readAll()).object();
        const auto state = result.value("state").toString();
        const int count = result.value("steps").toInt();
        if (state == "training") {
            status->setText(QStringLiteral("辞書登録済み・TTT学習中（%1回）…").arg(count));
            return;
        }
        if (state == "done") {
            status->setText(result.value("matched").toBool()
                ? QStringLiteral("辞書・TTT反映済み（%1回）。モデル単体でも指定した単語を生成できました。").arg(count)
                : QStringLiteral("辞書登録済み（学習%1回）。モデル単体の生成は未達のため重みの変更は戻しました。登録語は変換候補に出せます。").arg(count));
        } else if (state == "error") {
            status->setText(QStringLiteral("学習に失敗しました：%1").arg(result.value("message").toString()));
        } else return;
        timer->stop();
        submit->setEnabled(true);
    });
}

void DictionaryTabController::setContext(const TabContext& context) {
    context_ = context;
}
void DictionaryTabController::loadFromConfig() {}
void DictionaryTabController::saveToConfig() {}

}  // namespace hazkey::settings
