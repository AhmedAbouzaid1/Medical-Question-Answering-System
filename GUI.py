
from MedicalKG_.MedicalKBQA.question_classifier import QuestionClassifier
from MedicalKG_.MedicalKBQA.question_parser import QuestionPaser
from answer_search import *
from fetch import tagsanswer, check_similarity
from tkinter import *
import time

'''问答类'''


class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        answer = "Hello, I am XiaoMar Medical Assistant, I hope I can help you. If I don't answer it, I suggest you consult a professional doctor. I wish you a great body!"
        res_classify = self.classifier.classify(sent)
        print(res_classify)
        if not res_classify:
            answer, index = tagsanswer(sent)
            print("sim: ", answer)
            return answer

        res_sql = self.parser.parser_main(res_classify)
        print(res_sql)
        final_answers = self.searcher.search_main(res_sql)
        print(final_answers)

        if not final_answers:
            answer, index = tagsanswer(sent)
            print("sim: ", answer)
            return answer
        else:
            pred, prob = check_similarity(sent, final_answers[0])
            print(pred, " ", prob)

            if (pred == "contradiction"):
                answer, index = tagsanswer(sent)
                print("sim: ", answer)
                return answer

            return '\n'.join(final_answers)


def main():
    handler = ChatBotGraph()

    def retrieve_input():  # 发送消息
        strMsg = 'User:' + time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime()) + '\n'
        chatWindow.tag_config('quest', foreground="green")
        chatWindow.insert(END, strMsg, 'quest')
        chatWindow.insert(END, messageWindow.get('0.0', END))
        text = messageWindow.get('0.0', END)
        messageWindow.delete('0.0', END)
        print("text is " + text)

        text2 = handler.chat_main(text) + '\n'
        strMsg2 = 'Medical Assistant:' + time.strftime("%Y-%m-%d %H:%M:%S",
                                             time.localtime()) + '\n'
        chatWindow.insert(END, strMsg2, 'quest')
        chatWindow.insert(END, text2)
    #
    # def cancelMsg():  # 取消消息
    #     txtMsg.delete('0.0', END)

    # def retrieve_input():
    #     input = messageWindow.get("1.0", END)
    #     print(input)
    #     # messageWindow.delete('0.0', END)
    #     # answer = handler.chat_main(input)
    #     text2 = handler.chat_main(input) + '\n '
    #     writeanswer(text2)

    # def writeanswer(answer):
    #     strMsg = 'User:' + time.strftime("%Y-%m-%d %H:%M:%S",
    #                                      time.localtime()) + '\n'
    #     chatWindow.insert(END, strMsg, 'greencolor')
    #     chatWindow.insert(END, messageWindow.get("1.0", END))
    #     text = messageWindow.get('0.0', END)
    #     messageWindow.delete('0.0', END)
    #     strMsg2 = 'Medical Assistant:' + time.strftime("%Y-%m-%d %H:%M:%S",
    #                                          time.localtime()) + '\n'
    #     chatWindow.insert(END, strMsg2, 'greencolor')
    #     chatWindow.insert(END, answer)


    root = Tk()

    frmchat = Frame(width=50, height=8, bg='white')
    frmdisp = Frame(width=500, height=150, bg='white')
    frmbtn = Frame(width=500, height=30)

    root.title("Medical Assistant")
    root.geometry("500x500")
    root.configure(bg='light gray')
    root.resizable(width=TRUE, height=TRUE)
    chatWindow = Text(root, bd=1, bg="white", width="50", height="8", font=("Arial", 12))
    chatWindow.place(x=10, y=10, height=380, width=470)
    messageWindow = Text(root, bd=0, bg="white", width="30", insertbackground= "black", height="4", font=("Arial", 12), foreground="black")
    messageWindow.place(x=128, y=400, height=88, width=352)
    scrollbar = Scrollbar(root, command=chatWindow.yview)
    scrollbar.place(x=485, y=5, height=380)
    Button1 = Button(root, text="Send",
                    bd=0, bg="#0080ff",foreground='white', font=("Arial", 12), command= lambda: retrieve_input())
    Button1.place(x=10, y=400, height=88, width= 110)

    root.mainloop()

if __name__ == '__main__':
    main()