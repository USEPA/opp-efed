import os
import time
from tkinter import *
from tkinter import filedialog
import read_summary_file

project_path = os.path.dirname(os.path.abspath(__file__))


class TextEntry(object):
    def __init__(self, label="", width=100, sticky='w', default=None):
        global row
        self.label = Label(window, text=label)
        self.entry = Entry(window, width=width)
        self.label.grid(column=0, row=row, sticky=sticky, padx=10)
        self.entry.grid(column=0, row=row + 1, padx=10)
        if default is not None:
            self.update(default)
        row += 10

    def update(self, val):
        self.entry.delete(0, 'end')
        self.entry.insert(0, val)

    @property
    def value(self):
        out_val = self.entry.get()
        return out_val


class PathEntry(TextEntry):
    def __init__(self, label="", width=100, sticky='w', directory=False, ext=None, default=None):
        super(PathEntry, self).__init__(label, width, sticky, default)
        self.button = Button(window, text="Browse", command=lambda: self.browse(directory, ext))
        self.button.grid(column=1, row=row - 9, padx=0)

    def browse(self, directory, ext):
        if directory:
            path = filedialog.askdirectory(initialdir=project_path, defaultextension=ext)
        else:
            path = filedialog.askopenfilename(initialdir=project_path)
        self.update(path)


class NumericalEntry(TextEntry):
    def __init__(self, label="", width=100, sticky='w', single=False, dtype=float, default=None):
        super(NumericalEntry, self).__init__(label, width, sticky, default)
        self.single = single
        self.dtype = dtype

    @property
    def value(self):
        out_val = self.entry.get()
        if self.single:
            out_val = self.dtype(out_val)
        else:
            out_val = [self.dtype(val.strip()) for val in out_val.split(",")]
        return out_val


class ResultsWindow(object):
    def __init__(self):
        global row
        self.label = Label(window, text="Results")
        self.label.grid(column=0, row=row, sticky='nw', padx=10)
        self.text = Text(window, height=4, width=75)
        self.text.grid(column=0, row=row + 1)
        self.scroll = Scrollbar(window, command=self.text.yview)
        self.scroll.grid(column=1, row=row + 1, sticky='nsw')
        self.text['yscrollcommand'] = self.scroll.set
        row += 10

    def update(self, msg):
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', msg)


class SubmitButton(object):
    def __init__(self):
        global row
        run_button = Button(window, text="Run", command=self.run)
        run_button.grid(column=0, row=row)
        row += 10

    def run(self):
        start = time.time()
        outfiles = read_summary_file.main(
            *[f.value for f in (scenario_input, pwc_output, output_dir, sel_pcts, sel_window)])
        run_time = time.time() - start
        msg = "Successfully completed in {} seconds".format(int(run_time))
        msg += "\nCreated files:\n\t" + "\n\t".join(map(str, outfiles))
        results.update(msg)


# Initialize window
row = 0
window = Tk()
window.title("PWC Batch Postprocessor")
window.geometry('700x350')

scenario_input = PathEntry("Scenario File", ext=".csv")
pwc_output = PathEntry("PWC Output File")
output_dir = PathEntry("Output Location", directory=True)
sel_pcts = NumericalEntry("Selection Percentiles", default="50, 75, 90, 95")
sel_window = NumericalEntry("Selection Window", single=True, default="0.1")

SubmitButton()
results = ResultsWindow()

window.mainloop()
