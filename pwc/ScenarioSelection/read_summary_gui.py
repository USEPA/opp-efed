import os
import time
from tkinter import *
from tkinter import filedialog
import read_summary_file

project_path = os.path.dirname(os.path.abspath(__file__))


class PathEntry(object):
    def __init__(self, text=None, row=0, width=100, sticky='w', directory=False, ext=None, default=None):
        self.label = Label(window, text=text)
        self.entry = Entry(window, width=width)
        self.button = Button(window, text="Browse", command=lambda: self.browse(directory, ext))
        self.label.grid(column=0, row=row, sticky=sticky, padx=10)
        self.entry.grid(column=0, row=row + 1, padx=10)
        self.button.grid(column=1, row=row + 1, padx=10)
        if default is not None:
            self.update(default)

    def browse(self, directory, ext):
        if directory:
            path = filedialog.askdirectory(initialdir=project_path, defaultextension=ext)
        else:
            path = filedialog.askopenfilename(initialdir=project_path)
        self.update(path)

    @property
    def value(self):
        out_val = self.entry.get()
        return out_val

    def update(self, val):
        self.entry.delete(0, 'end')
        self.entry.insert(0, val)


class ResultsWindow(object):
    def __init__(self, row):
        self.text = Text(window, height=6, width=70, padx=20, pady=10)
        self.text.grid(column=0, row=row)
        self.scroll = Scrollbar(window, command=self.text.yview)
        self.scroll.grid(column=1, row=row, sticky='nsew')
        self.text['yscrollcommand'] = self.scroll.set

    def update(self, msg):
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', msg)


class SubmitButton(object):
    def __init__(self, row):
        self.row = row
        run_button = Button(window, text="Run", command=self.run)
        run_button.grid(column=0, row=row)

    def run(self):
        start = time.time()
        outfiles = read_summary_file.main(scenario_input.value, pwc_output.value, output_dir.value)
        run_time = time.time() - start
        msg = "Successfully completed in {} seconds".format(int(run_time))
        msg += "\nCreated files:\n\t" + "\n\t".join(map(str, outfiles))
        results.update(msg)


# Initialize window
window = Tk()
window.title("PWC Batch Postprocessor")
window.geometry('700x350')

scenario_input = PathEntry("Scenario File", ext=".csv")
pwc_output = PathEntry("PWC Output File", 10)
output_dir = PathEntry("Output Location", 20, directory=True)
SubmitButton(30)
results = ResultsWindow(40)

window.mainloop()
