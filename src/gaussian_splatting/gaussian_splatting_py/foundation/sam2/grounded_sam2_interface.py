import os
from typing import Optional, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import supervision as sv
from torchvision.ops import box_convert

import sys
sys.path.append("Grounded-SAM-2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from PIL import Image


def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer

def codet_setup_cfg(codet_config_file, codet_confidence_threshold, codet_opts):
    sys.path.insert(0, 'CoDet/')
    sys.path.insert(0, 'CoDet/third_party/CenterNet2/')
    from centernet.config import add_centernet_config
    from codet.config import add_codet_config
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_codet_config(cfg)
    cfg.merge_from_file(codet_config_file)
    cfg.merge_from_list(codet_opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = codet_confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = codet_confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = codet_confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    # if not args.codet_pred_all_class:
    #     cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.MODEL.DETR.CAT_FREQ_PATH = 'CoDet/datasets/metadata/lvis_v1_train_cat_info.json'
    cfg.MODEL.ROI_BOX_HEAD.DETECTION_WEIGHT_PATH = 'CoDet/datasets/metadata/lvis_v1_clip_a+cname.npy'
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = 'codet_files/lvis_v1_train_norare_cat_info.json'
    cfg.freeze()
    return cfg

def codet_get_clip_embeddings(vocabulary, prompt='a '):
    from codet.modeling.text.text_encoder import build_text_encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

class GroundedSAM2Interface:
    def __init__(self, bbox_model, device: Optional[Any]=None, debug: bool = False, debug_dir: str = 'debug',
        sam2_checkpoint="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_dino_config="Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth",
        codet_config_file="CoDet/configs/CoDet_OVLVIS_SwinB_4x_ft4x.yaml",
        codet_confidence_threshold=0.15,
        codet_opts=[]
        ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.debug = debug
        self.debug_dir = debug_dir
        self.viz_i = 0
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            

        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = sam2_model_config

        # build SAM2 image predictor
        model_cfg = sam2_model_config
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        if not (bbox_model == 'clip'):
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        self.bbox_model = bbox_model
        # build grounding dino model
        if bbox_model == 'gdino':
            from grounding_dino.groundingdino.util.inference import load_model
            import groundingdino.datasets.transforms as T
            self.grounding_model = load_model(
                model_config_path=grounding_dino_config, 
                model_checkpoint_path=grounding_dino_checkpoint,
                device=self.device
            )
            self.img_transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
            )
        elif bbox_model == 'florence':
            from transformers import AutoProcessor, AutoModelForCausalLM
            FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
            self.florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
            self.florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)
        elif bbox_model == 'clip':
            import clip
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.sam2_mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
        elif bbox_model == 'codet':
            from detectron2.data import MetadataCatalog
            from detectron2.engine.defaults import DefaultPredictor
            cfg = codet_setup_cfg(codet_config_file, codet_confidence_threshold, codet_opts)
            self.metadata = MetadataCatalog.get("__unused")
            self.codet_predictor = DefaultPredictor(cfg)

        

    def get_mask_text_prompt(self, img: np.array, text_prompt: str, box_threshold: float=0.35, text_threshold: float = 0.25):
        if self.bbox_model == 'clip':
            text = clip.tokenize(text_prompt).to(self.device)
            masks = self.sam2_mask_generator.generate(img)
            best_sim = 0
            best_mask = None
            for mask in masks:
                bbox = mask['bbox']
                img_c = img[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]
                # plt.imshow(img*np.tile(mask['segmentation'][:,:,None],(1,1,3)))
                # plt.show()
                # plt.imshow(img_c)
                # plt.show()

                image = self.clip_preprocess(Image.fromarray(img_c)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image)
                    text_features = self.clip_model.encode_text(text)
                    
                    
                    sim = (image_features@text_features.t()).squeeze().item()
                    if sim > best_sim:
                        best_mask=mask['segmentation'].copy()
                        best_sim = sim

            return best_mask


        self.sam2_predictor.set_image(img)
        image_pil = Image.fromarray(img)

        if self.bbox_model=='gdino':
            from grounding_dino.groundingdino.util.inference import predict
            image_transformed, _ = self.img_transform(image_pil, None)
            if text_prompt.find(' .') == -1:
                text = text_prompt + ' .'
            else:
                text = text_prompt
            filter_classes = False

            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image_transformed,
                caption=text,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            # process the box prompt for SAM 2
            h, w, _ = img.shape
            boxes = boxes * torch.tensor([w, h, w, h],device='cpu')
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        elif self.bbox_model=='florence':
            results = run_florence2("<OPEN_VOCABULARY_DETECTION>", 
                text_prompt, self.florence2_model, self.florence2_processor, image_pil)
            
            results = results["<OPEN_VOCABULARY_DETECTION>"]
            # parse florence-2 detection results
            input_boxes = np.array(results["bboxes"])
            labels = results["bboxes_labels"]
            class_ids = np.array(list(range(len(labels))))
        elif self.bbox_model == 'codet':
            predictions = self.codet_predictor(img)

            scores = predictions['instances'].scores
            bboxes = predictions['instances'].pred_boxes.tensor

            if scores.numel() == 0:
                return None

            best_idx = torch.argmax(scores.squeeze())
            input_boxes = bboxes[best_idx].cpu().numpy()

        if input_boxes.shape[0] == 0:
            print(f"Failed to detect masks for object {text_prompt}")
            return None

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        if self.bbox_model=='gdino':
            confidences = confidences.numpy().tolist()
        else:
            confidences = scores.tolist()

        if len(confidences)==0:
            print(f"Failed to detect masks for object {text_prompt}")
            return None

        best_idx = np.argmax(confidences)

        mask = masks[best_idx].astype(bool)

        if self.debug:
            class_ids = np.array(list(range(len(labels))))
            detections = sv.Detections(
                    xyxy=input_boxes,  # (n, 4)
                    mask=masks.astype(bool),  # (n, h, w)
                    class_id=class_ids
                )

            labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence
                    in zip(labels, confidences)
                ]

            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

            cv2.imwrite(os.path.join(f'{self.debug_dir}',f'grounded_sam2_text_{self.viz_i}.jpg'), annotated_frame)
            self.viz_i += 1

        return mask

    def get_mask_point_prompt(self, img: np.array, point_prompt: np.ndarray, input_label: np.ndarray = np.array([1])):
        self.sam2_predictor.set_image(img)
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=point_prompt,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        if self.debug:
            scores = scores[sorted_ind]
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                show_mask(mask, plt.gca(), borders=False)
                if point_prompt is not None:
                    assert input_labels is not None
                    show_points(point_prompt, input_labels, plt.gca())
                if len(scores) > 1:
                    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                plt.axis('off')
                plt.savefig(os.path.join(f'{self.debug_dir}',f'grounded_sam2_point_{self.viz_i}_{i}.jpg'))
            self.viz_i += 1
        return masks[0]

    def save_masks_text(self, text_prompt, data_dir, args=None):
        import glob
        os.makedirs(os.path.join(data_dir,'masks'),exist_ok=True)

        if self.bbox_model == 'codet':
            sys.path.insert(0, 'CoDet/')
            from codet.modeling.utils import reset_cls_infer

            self.metadata.thing_classes = text_prompt.split(',')
            classifier = codet_get_clip_embeddings(self.metadata.thing_classes)

            num_classes = len(self.metadata.thing_classes)
            reset_cls_infer(self.codet_predictor.model, classifier, num_classes)
        
        
        for fn in sorted(glob.glob(os.path.join(data_dir,'jpg_images/*.jpg'))):
            image = cv2.imread(fn)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = self.get_mask_text_prompt(image, text_prompt)
            if not mask is None:
                cv2.imwrite(fn.replace('jpg_images','masks').replace('jpg','png'), mask.astype(np.uint8)*255)
            else:
                cv2.imwrite(fn.replace('jpg_images','masks').replace('jpg','png'), 
                    np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8))

    def save_masks_video(self, text_prompt, data_dir, args=None):
        import glob
        from sam2.build_sam import build_sam2_video_predictor

        os.makedirs(os.path.join(data_dir,'masks'),exist_ok=True)

        if self.bbox_model == 'codet':
            sys.path.insert(0, 'CoDet/')
            from codet.modeling.utils import reset_cls_infer

            self.metadata.thing_classes = text_prompt.split(',')
            classifier = codet_get_clip_embeddings(self.metadata.thing_classes)

            num_classes = len(self.metadata.thing_classes)
            reset_cls_infer(self.codet_predictor.model, classifier, num_classes)
        
        fn = sorted(glob.glob(os.path.join(data_dir,'jpg_images/*.jpg')))[0]
        image = cv2.imread(fn)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        box_threshold = 0.35
        text_threshold = 0.25
        mask = self.get_mask_text_prompt(image, text_prompt, box_threshold, text_threshold)

        if self.bbox_model=='gdino':
            while mask is None:
                box_threshold *= 0.9
                text_threshold *= 0.9
                mask = self.get_mask_text_prompt(image, text_prompt, box_threshold, text_threshold)
        else:
            if mask is None:
                raise Exception("No mask found")
        video_predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint)
        inference_state = video_predictor.init_state(video_path=os.path.join(data_dir,'jpg_images'))

        labels = np.ones((1), dtype=np.int32)
        ann_frame_idx = 0
        object_id = 0
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask
        )
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        for k,v in video_segments.items():
            cv2.imwrite(os.path.join(data_dir,'masks',f'{k:04d}.png'), v[0][0].astype(np.uint8)*255)

if __name__ == "__main__":    
    import argparse

    arg = argparse.ArgumentParser()
    arg.add_argument('--data_dir', type=str, help='Path to the color image')
    arg.add_argument('--text_prompt', type=str, help='Class to segment')
    arg.add_argument('--bbox_model', type=str, choices=['gdino','florence','clip', 'codet'])
    arg.add_argument('--video_mode', action='store_true')
    arg.add_argument('--sam2_checkpoint', type=str, default="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    arg.add_argument('--sam2_model_config', type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    arg.add_argument('--grounding_dino_config', type=str, default="Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    arg.add_argument('--grounding_dino_checkpoint', type=str, default="Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth")
    arg.add_argument('--codet_config_file', type=str, default="CoDet/configs/CoDet_OVLVIS_SwinB_4x_ft4x.yaml")
    arg.add_argument('--codet_confidence_threshold', type=float,default=0.15)
    arg.add_argument(
        "--codet_opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    args = arg.parse_args()

    gs = GroundedSAM2Interface(args.bbox_model, sam2_checkpoint=args.sam2_checkpoint, sam2_model_config=args.sam2_model_config, 
        grounding_dino_config=args.grounding_dino_config, grounding_dino_checkpoint=args.grounding_dino_checkpoint, 
        codet_config_file=args.codet_config_file,codet_confidence_threshold=args.codet_confidence_threshold, codet_opts=args.codet_opts)

    if args.video_mode:
        gs.save_masks_video(args.text_prompt, args.data_dir)
    else:
        gs.save_masks_text(args.text_prompt, args.data_dir)