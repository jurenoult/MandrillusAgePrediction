import logging
import os

import napari
import numpy as np
import skimage.io as io
from PIL import Image
from qtpy.QtWidgets import QPushButton
from skimage.transform import resize

from .dataset import SAM2LikeTransform

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# os.environ["QT_QPA_PLATFORM"] = "offscreen"
# os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"


def infer_im(im_path, gt_path, gt_folder, transforms, model):
    im = Image.open(im_path).convert("RGB")

    # Prepare image tensor
    im_tensor = transforms(im)
    # Put the tensor on the same device as the model
    im_tensor = im_tensor.to(model.device)

    prediction = model(im_tensor.unsqueeze(0))

    # From tensor to numpy array
    prediction = prediction.detach().cpu().squeeze(0).numpy()

    # Resize
    prediction = resize(
        prediction,
        (2, im.size[1], im.size[0]),
        order=1,
        mode="reflect",
        anti_aliasing=True,
    )

    prediction = np.argmax(prediction, axis=0)
    prediction = (prediction * 255).astype(np.uint8)

    # Save groundtruth and reload
    if not os.path.exists(gt_folder):
        os.makedirs(gt_folder)
    io.imsave(gt_path, prediction)


class KEYBINDS(object):
    PREV_SAMPLE = "l"
    NEXT_SAMPLE = "m"
    REAL_ERROR = "e"
    NEXT_NO_GT = "j"


class ImageViewer(object):
    def __init__(
        self,
        model,
        all_paths,
    ):
        self.model = model
        self.all_paths = all_paths
        self.transforms = SAM2LikeTransform(is_train=False)
        self.current_idx = 0

        self.img = np.zeros((224, 224, 3), dtype=np.uint8)
        self.label = np.zeros((224, 224, 1), dtype=np.uint8)
        self.create_viewer()

        # Initialise empty points layers
        self.prev_button = QPushButton("(L) Previous")
        self.next_button = QPushButton("(M) Next")
        self.error_button = QPushButton("(E)rror")

        self.viewer.window.add_dock_widget(self.prev_button, area="left")
        self.viewer.window.add_dock_widget(self.next_button, area="left")
        self.viewer.window.add_dock_widget(self.error_button, area="left")

        self.prev_button.clicked.connect(self.prev_sample)
        self.next_button.clicked.connect(self.next_sample)
        self.error_button.clicked.connect(self.delete_label)

        self.load(self.current_idx)
        self.start()

    def delete_label(self):
        # Delete label image
        im_folder, gt_folder, im_path, gt_path = self.all_paths[self.current_idx]
        if os.path.exists(gt_path):
            os.remove(gt_path)

        if "segmentation" not in self.viewer.layers:
            self.label_layer = self.viewer.add_image(
                self.label, name="segmentation", colormap="blue", opacity=0.3
            )

        # Update display in napari by reloading
        self.load(self.current_idx)

    def next_no_gt(self):
        for i in range(self.current_idx + 1, len(self.all_paths)):
            im_folder, gt_folder, im_path, gt_path = self.all_paths[i]
            if not os.path.exists(gt_path):
                print(f"GT path does not exists = {gt_path}")
                self.current_idx = i
                break

        # Update display in napari by reloading
        self.load(self.current_idx)

    def prev_sample(self):
        self.current_idx = self.current_idx - 1
        if self.current_idx < 0:
            self.current_idx = len(self.all_paths) - 1
        self.load(self.current_idx)

    def next_sample(self, amount=1):
        self.current_idx = (self.current_idx + amount) % len(self.all_paths)
        self.load(self.current_idx)

    def update_label(self, idx):
        status_message = f"image = {idx}/{len(self.all_paths)}"
        self.viewer.text_overlay.text = status_message

    def load(self, idx):
        im_folder, gt_folder, im_path, gt_path = self.all_paths[idx]
        print(f"Loading image idx=[{idx}]: {im_path}")

        self.update_label(idx)

        # Load image
        im = io.imread(im_path, as_gray=False)
        label = np.zeros_like(im[..., 0])
        if os.path.exists(gt_path):
            label = io.imread(gt_path, as_gray=True)

        # self.image_layer.data = np.transpose(im, (-1, 0, 1))
        self.image_layer.data = im
        self.label_layer.data = label

        # current_step = im.shape[0] // 2
        # self.viewer.dims.set_current_step(0, current_step)
        self.viewer.camera.center = (
            im.shape[0] // 2,
            im.shape[1] // 2,
            # im.shape[2] // 2,
        )
        self.viewer.camera.zoom = 1

    def infer(self):
        im_folder, gt_folder, im_path, gt_path = self.all_paths[self.current_idx]
        infer_im(im_path, gt_path, gt_folder, self.transforms, self.model)
        self.load(self.current_idx)

    def create_viewer(self):
        self.viewer = napari.Viewer()

        self.image_layer = self.viewer.add_image(self.img, name="image", rgb=True)
        self.label_layer = self.viewer.add_image(
            self.label, name="segmentation", colormap="blue", opacity=0.3
        )

        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.font_size = 20
        self.viewer.text_overlay.color = (1.0, 1.0, 1.0, 1.0)

        @self.viewer.bind_key(KEYBINDS.PREV_SAMPLE)
        @self.viewer.bind_key("Left")
        def do_prev_sample(viewer):
            self.prev_sample()

        @self.viewer.bind_key(KEYBINDS.NEXT_SAMPLE)
        @self.viewer.bind_key("Right")
        def skip_sample(viewer):
            self.next_sample()

        @self.viewer.bind_key(KEYBINDS.REAL_ERROR)
        @self.viewer.bind_key("Delete")
        def set_real_error(viewer):
            self.delete_label()

        @self.viewer.bind_key("Return")
        @self.viewer.bind_key("Enter")
        def perform_infer(viewer):
            self.infer()

        @self.viewer.bind_key("Shift-Left")
        def prev_n_sample(viewer):
            self.next_sample(-50)

        @self.viewer.bind_key("Shift-Right")
        def next_n_sample(viewer):
            self.next_sample(50)

        @self.viewer.bind_key(KEYBINDS.NEXT_NO_GT)
        def next_no_gt_(viewer):
            self.next_no_gt()

    def start(self):
        napari.run()
        logger.info("Napari viewer started !")
