import argparse
import cv2
import os
import time
from ditod import add_vit_config
import torch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from ditod.VGTTrainer import DefaultPredictor


def main():
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    parser.add_argument(
        "--image_root",
        help="Path to image folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--grid_root",
        help="Path to grid folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_root",
        help="Name of the output visualization folder.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="Name of the dataset. Options: publaynet, docbank, D4LA, doclaynet",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="Path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # Get paths to images
    image_paths = [
        os.path.join(args.image_root, f)
        for f in os.listdir(args.image_root)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Initialize the configuration
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)

    # Add model weights URL to config
    cfg.merge_from_list(args.opts)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Define model
    predictor = DefaultPredictor(cfg)

    # Metadata setup based on dataset
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if args.dataset == "publaynet":
        md.set(thing_classes=["text", "title", "list", "table", "figure"])
    elif args.dataset == "docbank":
        md.set(
            thing_classes=[
                "abstract",
                "author",
                "caption",
                "date",
                "equation",
                "figure",
                "footer",
                "list",
                "paragraph",
                "reference",
                "section",
                "table",
                "title",
            ]
        )
    elif args.dataset == "D4LA":
        md.set(
            thing_classes=[
                "DocTitle",
                "ParaTitle",
                "ParaText",
                "ListText",
                "RegionTitle",
                "Date",
                "LetterHead",
                "LetterDear",
                "LetterSign",
                "Question",
                "OtherText",
                "RegionKV",
                "Regionlist",
                "Abstract",
                "Author",
                "TableName",
                "Table",
                "Figure",
                "FigureName",
                "Equation",
                "Reference",
                "Footnote",
                "PageHeader",
                "PageFooter",
                "Number",
                "Catalog",
                "PageNumber",
            ]
        )
    elif args.dataset == "doclaynet":
        md.set(
            thing_classes=[
                "Caption",
                "Footnote",
                "Formula",
                "List-item",
                "Page-footer",
                "Page-header",
                "Picture",
                "Section-header",
                "Table",
                "Text",
                "Title",
            ]
        )

    # Process each image
    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        if args.dataset in ("D4LA", "doclaynet", "docbank"):
            grid_path = os.path.join(args.grid_root, image_name + ".pkl")
        else:
            grid_path = os.path.join(args.grid_root, image_name + ".pdf.pkl")

        output_file_name = os.path.join(args.output_root, image_name + ".jpg")

        print("Attempting to load image from", image_path)

        # Load the image
        img = cv2.imread(image_path)

        # Check if the image was loaded correctly
        if img is None:
            print(
                "Error: The image could not be loaded. Please check the file path and format."
            )
            continue
        else:
            print("Image loaded successfully. Image shape:", img.shape)

        # Run inference and measure time
        start_time = time.time()
        output = predictor(img, grid_path)
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time for {image_path}: {inference_time:.2f} seconds")

        # Filter instances based on confidence threshold
        confidence_threshold = 0.8
        instances = output["instances"]
        high_confidence_idxs = instances.scores > confidence_threshold
        filtered_instances = instances[high_confidence_idxs]

        # Visualize the filtered instances
        v = Visualizer(
            img[:, :, ::-1], md, scale=1.0, instance_mode=ColorMode.SEGMENTATION
        )
        result = v.draw_instance_predictions(filtered_instances.to("cpu"))
        result_image = result.get_image()[:, :, ::-1]

        print("Saving result to", output_file_name)
        # Save the result image
        if not os.path.exists(args.output_root):
            os.makedirs(args.output_root)
        cv2.imwrite(output_file_name, result_image)


if __name__ == "__main__":
    main()
