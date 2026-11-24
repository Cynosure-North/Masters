import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import to_rgba_array


plt.rcParams['axes3d.mouserotationstyle'] = 'azel'

POINT_NAMES = [
	"L_wrist",
	"L_vec",
	"L_thumb",
	"L_index",
	"L_middle",
	"L_ring",
	"L_little",
	"R_wrist",
	"R_vec",
	"R_thumb",
	"R_index",
	"R_middle",
	"R_ring",
	"R_little",
]


def vis_raw():
	first_file = Path(r"H:\Research_Data\Morgana North(MHIT)\phase1\session_32d94b77\legs\a fox is a very smart animal")

	# data stores per-frame deltas of shape (n_frames, n_points, 3)
	record = eval(first_file.read_text(), {"__builtins__": {}, "array": np.array})
	# I collected the frame timestamp, but I'm not using it at the moment so only grab the frames
	data = np.asarray([frame for _, frame in record]).reshape(-1, 14, 3)

	# # Integrate deltas across frames to get 3D coordinates
	# coords = np.cumsum(data, axis=0)
	coords = data

	num_steps = coords.shape[0]
	t_values = np.linspace(0, 10, num_steps)

	# Indices 2 and 9 are vectors originating at reconstructed indices 1 and 8
	# origin_indices = [0, 7]
	# vector_indices = [1, 8]
	point_mask = np.ones(14, dtype=bool)
	point_mask[[0,1,7,8]] = False

	pt_indices = np.flatnonzero(point_mask)
	pt_labels = [POINT_NAMES[i] for i in pt_indices]
	pt_colors = ["blue" if idx < 7 else "red" for idx in pt_indices]
	pt_rgba = to_rgba_array(pt_colors)

	pt_coords = coords[:, point_mask, :]
	x_pts, z_pts, y_pts = -pt_coords[:, :, 0], pt_coords[:, :, 1], pt_coords[:, :, 2]

	# vec_origins = coords[:, origin_indices, :]  # shape: (n_frames, 2, 3)
	# vec_dirs = coords[:, vector_indices, :]        # per-frame delta vectors: (n_frames, 2, 3)

	# 2. Set up the figure and 3D axis
	fig = plt.figure(figsize=(8, 7))
	plt.subplots_adjust(bottom=0.2)
	ax = fig.add_subplot(111, projection="3d")

	# def draw_vectors(step):
	# 	ox = vec_origins[step, :, 0]
	# 	oy = vec_origins[step, :, 2]
	# 	oz = vec_origins[step, :, 1]
	# 	u = vec_dirs[step, :, 0]
	# 	v = vec_dirs[step, :, 2]
	# 	w = vec_dirs[step, :, 1]
	# 	return ax.quiver(
	# 		-ox, oy, oz, -u, v, -w,
	# 		color="black",
	# 		linewidth=2,
	# 		normalize=False,
	# 		length=0.05,
	# 	)

	def draw_labels(step):
		return [
			ax.text(x_pts[step, i], y_pts[step, i], z_pts[step, i], f" {label}", fontsize=8)
			for i, label in enumerate(pt_labels)
			if label != "_"
		]

	# 3. Plot initial time step (t = 0)
	initial_step = 0
	scatter = ax.scatter(
		x_pts[initial_step],
		y_pts[initial_step],
		z_pts[initial_step],
		facecolors=pt_rgba,
		s=50,
	)
	# quiver = draw_vectors(initial_step)
	text_labels = draw_labels(initial_step)


	# Fix axis limits with equal scale across X, Y, and Z
	x_mid = 0.5 * (x_pts.min() + x_pts.max())
	y_mid = 0.5 * (y_pts.min() + y_pts.max())
	z_mid = 0.5 * (z_pts.min() + z_pts.max())

	max_range = 0.5 * max(
		x_pts.max() - x_pts.min(),
		y_pts.max() - y_pts.min(),
		z_pts.max() - z_pts.min(),
	)
	# Guard against degenerate/all-zero data where max_range == 0
	if max_range == 0:
		max_range = 1.0

	ax.set_xlim(x_mid - max_range, x_mid + max_range)
	ax.set_ylim(y_mid - max_range, y_mid + max_range)
	ax.set_zlim(z_mid - max_range, z_mid + max_range)
	ax.set_aspect("equal")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	title = ax.set_title(f"Time Step: {initial_step} (t = {t_values[initial_step]:.2f})")

	# 4. Create the slider axis and widget
	ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
	time_slider = Slider(
		ax=ax_slider,
		label="Time Step ",
		valmin=0,
		valmax=num_steps - 1,
		valinit=initial_step,
		valstep=1,
	)

	# 5. Define the update function
	def update(val):
		# nonlocal quiver, text_labels
		nonlocal text_labels
		step = int(time_slider.val)

		scatter._offsets3d = (x_pts[step], y_pts[step], z_pts[step])
		scatter.set_facecolors(pt_rgba)

		# quiver.remove()
		# quiver = draw_vectors(step)

		for txt in text_labels:
			txt.remove()
		text_labels = draw_labels(step)

		title.set_text(f"Time Step: {step} (t = {t_values[step]:.2f})")
		fig.canvas.draw_idle()

	time_slider.on_changed(update)

	plt.get_current_fig_manager().full_screen_toggle()
	plt.show()


if __name__ == "__main__":
	vis_raw()