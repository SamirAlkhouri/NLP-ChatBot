from tkinter import *
from bot_gui_data import retrieve_response, chatbot_name

FONT = "Merriweather 15"
FONT_BOLD = "Merriweather 16 bold"
BG_COLOR = "dark gray"
TEXT_COLOR = "black"
DIV_COLOR = "white"


class ChatBotApp:
    def __init__(self):
        self.window = Tk()
        self.main_window()

    def run(self):
        self.window.mainloop()

    def main_window(self):
        # window
        self.window.title("Chatty The ChatBot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=600, height=700, bg=BG_COLOR)

        # top label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Hello, I'm Chatty, let's have a chat!", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # bottom label
        bottom_label = Label(self.window, bg=DIV_COLOR, height=85)
        bottom_label.place(relwidth=1, rely=0.85)

        # text area
        self.text_area = Text(self.window, width=15, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                              font=FONT, padx=5, pady=5)
        self.text_area.place(relheight=0.76, relwidth=1, rely=0.08)
        self.text_area.configure(cursor="arrow", state=DISABLED)

        # divider
        line = Label(self.window, width=450, bg=DIV_COLOR, )
        line.place(relwidth=1, rely=0.08, relheight=0.012)

        # scrollbar
        scrollbar = Scrollbar(self.text_area)
        scrollbar.place(relheight=1, relx=0.98)
        scrollbar.configure(command=self.text_area.yview)

        # send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=DIV_COLOR,
                             command=lambda: self.enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.045, relwidth=0.22)

        # message box
        self.msg_entry = Entry(bottom_label, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.045, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self.enter_pressed)

    def enter_pressed(self, event):
        msg = self.msg_entry.get()
        self.send_message(msg, "You")

    def send_message(self, msg, sender):
        if not msg:
            return

        # sender messaging
        self.msg_entry.delete(0, END)
        sender_msg = f"{sender}: {msg}\n\n"
        self.text_area.configure(state=NORMAL)
        self.text_area.insert(END, sender_msg)
        self.text_area.configure(state=DISABLED)

        # bot messaging
        bot_msg = f"{chatbot_name}: {retrieve_response(msg)}\n\n"
        self.text_area.configure(state=NORMAL)
        self.text_area.insert(END, bot_msg)
        self.text_area.configure(state=DISABLED)
        self.text_area.see(END)


if __name__ == "__main__":
    application = ChatBotApp()
    application.run()
