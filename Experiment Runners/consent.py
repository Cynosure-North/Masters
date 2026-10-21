import tkinter as tk
from tkinter import ttk


def main() -> tuple[bool, bool] | None:
	"""Show the consent form and return optional permissions, or None if closed."""
	root = tk.Tk()
	root.title("Participant Consent")
	root.geometry("700x1000")
	root.resizable(False, False)
	root.configure(background="#f4f6f8")

	style = ttk.Style(root)
	style.configure("Page.TFrame", background="#f4f6f8")
	style.configure("Card.TFrame", background="#ffffff")
	style.configure("Title.TLabel", background="#f4f6f8", foreground="#172b4d", font=("Segoe UI", 18, "bold"))
	style.configure("Intro.TLabel", background="#f4f6f8", foreground="#44546a", font=("Segoe UI", 10))
	style.configure("Section.TLabel", background="#ffffff", foreground="#172b4d", font=("Segoe UI", 11, "bold"))
	style.configure("Form.TCheckbutton", background="#ffffff", foreground="#253858", font=("Segoe UI", 10))
	style.configure("Footer.TFrame", background="#f4f6f8")
	style.configure("Footer.TLabel", background="#f4f6f8", foreground="#44546a", font=("Segoe UI", 9))
	style.configure("Footer.Bold.TLabel", background="#f4f6f8", foreground="#172b4d", font=("Segoe UI", 9, "bold"))
	style.configure("Accept.TButton", font=("Segoe UI", 10, "bold"), padding=(18, 9))

	page = ttk.Frame(root, style="Page.TFrame", padding=(28, 24, 28, 0))
	page.pack(fill="both", expand=True)

	card = ttk.Frame(page, style="Card.TFrame", padding=(24, 20))
	card.pack(fill="both", expand=True)
	form = ttk.Frame(card, style="Card.TFrame")
	form.pack(fill="both", expand=True)

	mandatory_statements = [
		"I have been given a full explanation of this project and have had the opportunity to ask questions.",
		"I understand what is required of me if I agree to take part in the research.",
		"I understand that participation is voluntary and I may withdraw at any time without consequences. Withdrawal of participation will also include the withdrawal of any information I have provided should this remain possible.",
		"I understand that interview transcripts and questionnaire answers I provide will be kept confidential to the researcher. I understand that any published or reported results will not identify me.",
		"I understand that the research will collect and analyse data describing the movements and positions of my fingers and hands during the study.",
		"I understand that a thesis is a public document and will be available through the UC Library.",
		"I understand that all data collected for the study will be kept in locked and secure facilities and/or in password protected electronic form. I understand the transcript, questionnaire answers, and any finger motion data that has not been uploaded, will be destroyed after five years.",
		"I understand the risks associated with taking part and how they will be managed.",
		"I agree to being audio recorded. I understand how this recording will be stored and used.",
		"I understand that I can contact the researcher, Morgana North (morgana.north@pg.canterbury.ac.nz), or supervisors Prof. Dr. Stephan Lukosch (stephan.lukosch@canterbury.ac.nz) and Dr. Donald Degraen (donald.degraen@canterbury.ac.nz) for further information. If I have any complaints, I can contact the Chair of the University of Canterbury Human Research Ethics Committee, Private Bag 4800, Christchurch (email: human-ethics@canterbury.ac.nz).",
	]
	mandatory_vars = [tk.BooleanVar(value=False) for _ in mandatory_statements]
	optional_vars = [tk.BooleanVar(value=False), tk.BooleanVar(value=False)]
	wrapped_checks: list[tk.Checkbutton] = []

	ttk.Label(form, text="Tick each statement you agree with", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
	accept_button = ttk.Button(root, text="Accept and continue", style="Accept.TButton", state="disabled")

	def update_accept_button(*_args: object) -> None:
		all_confirmed = all(variable.get() for variable in mandatory_vars)
		accept_button.configure(state="normal" if all_confirmed else "disabled")

	def add_wrapped_check(statement: str, variable: tk.BooleanVar, command=None) -> None:
		checkbox = tk.Checkbutton(
			form,
			text=statement,
			variable=variable,
			command=command,
			wraplength=900,
			justify="left",
			anchor="w",
			background="#ffffff",
			activebackground="#ffffff",
			foreground="#253858",
			selectcolor="#ffffff",
			font=("Segoe UI", 10),
			borderwidth=0,
			highlightthickness=0,
		)
		checkbox.pack(anchor="w", fill="x", pady=5)
		wrapped_checks.append(checkbox)

	def resize_wrapped_checks(event: tk.Event) -> None:
		wraplength = max(420, event.width - 44)
		for checkbox in wrapped_checks:
			checkbox.configure(wraplength=wraplength)

	form.bind("<Configure>", resize_wrapped_checks)

	for variable, statement in zip(mandatory_vars, mandatory_statements):
		add_wrapped_check(statement, variable, update_accept_button)

	add_wrapped_check(
		"I consent to my de-identified finger motion data being shared in an open science data repository. I understand that, once anonymised data is publicly released, it may not be possible to withdraw them from the repository or from copies downloaded by others.",
		optional_vars[0],
	)
	add_wrapped_check(
		"I understand that my quotes may be included in the research outputs.",
		optional_vars[1],
	)

	result: tuple[bool, bool] | None = None

	def accept() -> None:
		nonlocal result
		result = (optional_vars[0].get(), optional_vars[1].get())
		root.destroy()

	accept_button.configure(command=accept)
	footer = ttk.Frame(page, style="Footer.TFrame", padding=(0, 14, 0, 18))
	footer.pack(fill="x")
	ttk.Label(
		footer,
		text="By clicking Accept, I agree to participate in this research project.",
		style="Intro.TLabel",
		wraplength=520,
		justify="left",
	).pack(side="left", fill="x", expand=True, padx=(0, 12))
	accept_button.pack(in_=footer, side="right")

	contact_footer = ttk.Frame(page, style="Footer.TFrame", padding=(0, 0, 0, 16))
	contact_footer.pack(fill="x")
	contact_details = ttk.Frame(contact_footer, style="Footer.TFrame")
	contact_details.pack(side="left", anchor="w")
	ttk.Label(contact_details, text="HIT Lab NZ", style="Footer.TLabel").pack(anchor="w")
	ttk.Label(contact_details, text="Phone: +64 3 369 2226 (HIT Lab NZ phone number)", style="Footer.TLabel").pack(anchor="w")
	ttk.Label(contact_details, text="HREC Ref: HREC 2026/55/LR", style="Footer.TLabel").pack(side="left")

	image = tk.PhotoImage(file=r"C:\Users\mno64\Thesis\Experiment Runners\uc_logo_smaller.png")
	photo = tk.Label(
		contact_footer,
		image=image,
		background="#f4f6f8",
		borderwidth=0,
		highlightthickness=0,
	)
	photo.pack(side="right")

	root.protocol("WM_DELETE_WINDOW", root.destroy)
	root.mainloop()
	return result

if __name__ == "__main__":
	main()