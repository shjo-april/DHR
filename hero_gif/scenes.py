"""Texts, numbers and panel lists of the hero GIF, read by build_gif.py. Every number is from the paper (see each note)."""

SCENES = {
 "title": "DHR",
 "badge": "ECCV 2024",
 "subtitle": "Dual Features-Driven Hierarchical Rebalancing in Inter- and Intra-Class Regions for Weakly-Supervised Semantic Segmentation",
 "tabs": ["1 · Problem", "2 · Dual features", "3 · Rebalancing", "4 · Results"],
 "problem": {
  "columns": [["image.jpg", "Image"], ["gt.png", "Ground truth"], ["toco.png", "ToCo"], ["mars.png", "MARS"], ["dhr.png", "DHR (Ours)"]],
  "rows": [
   {"id": "2007_007048", "tags": [["bottle", [128, 0, 128]], ["sheep", [128, 64, 0]], ["person", [192, 128, 128]]], "box": [0.0, 0.0934, 0.1928, 0.491]},
   {"id": "2009_004368", "tags": [["motorbike", [64, 128, 128]], ["person", [192, 128, 128]]], "box": [0.4144, 0.1441, 1.0, 0.9159]}
  ],
  "messages": [
   "Weakly-supervised segmentation (WSS) learns masks from image-level class labels only",
   "In recent WSS masks, a minor class next to a larger one vanishes: no bottle, almost no rider",
   "DHR restores the vanished minor classes"
  ],
  "stat": "Adjacent regions make up 35% of the total area on PASCAL VOC and 75% on MS COCO (paper, Sec. 1)",
  "note": "Paper Fig. 2 · pseudo labels of two PASCAL VOC 2012 training images from ToCo, MARS and DHR · boxes mark the adjacent minor class"
 },
 "features": {
  "panels": [
   ["paper/fig3_a.png", "(a) USS advantage", "USS features separate inter-class regions", "classes with dissimilar appearances, e.g. person vs. motorbike"],
   ["paper/fig3_b.png", "(b) WSS advantage", "WSS features separate intra-class regions", "visually similar classes, e.g. dog vs. cat"]
  ],
  "messages": [
   "Unsupervised (USS) features tell the rider from the motorbike, while weakly-supervised (WSS) features do not",
   "WSS features, trained with class labels, tell the cat from the dog, while USS features do not"
  ],
  "note": "Paper Fig. 3 · pixel-wise cosine similarity maps (heatmaps) of WSS and USS features for the marked class points",
  "hierarchy": "paper/fig4.png",
  "hierarchy_message": "USS features group the inter classes (vehicle, animal, indoor), then WSS features separate the intra classes of each group",
  "hierarchy_note": "Paper Fig. 4 · left: the correlation of USS features groups the inter classes · right: hierarchical clustering with USS features, then WSS features"
 },
 "method": {
  "rows": [
   {"id": "000000112673", "tags": [["table", [192, 64, 192]], ["chair", [192, 64, 64]], ["bottle", [64, 0, 64]], ["cup", [64, 128, 64]], ["bowl", [64, 128, 192]], ["knife", [64, 0, 192]], ["sandwich", [128, 64, 64]], ["person", [128, 0, 0]]]},
   {"id": "000000460202", "tags": [["dog", [128, 64, 0]], ["cat", [0, 64, 0]], ["person", [128, 0, 0]], ["backpack", [192, 64, 0]]]}
  ],
  "steps": [
   ["input.png", "INPUT", "Input WSS mask", "from RSEPM with MARS", 52.6, "The input WSS mask misses the minor classes next to larger ones"],
   ["step1.png", "STEP 1", "OT-based seed initialization", "for vanished minor classes", 55.1, "Step 1 recovers seeds of the vanished classes with optimal transport on the CAMs"],
   ["step2.png", "STEP 2", "USS rebalancing", "for inter-class regions", 56.8, "Step 2 propagates the seeds with USS features and separates inter-class regions"],
   ["step3.png", "STEP 3", "WSS rebalancing", "for intra-class regions", 58.6, "Step 3 applies WSS features and separates intra-class regions: bottle vs. cup, dog vs. cat"]
  ],
  "summary": "Pseudo-mask mIoU on the MS COCO 2014 training set: 52.6 → 58.6 (+6.0)",
  "note": "Paper Fig. 5 (top image) and Table 6 · masks of two MS COCO 2014 training images after each step · numbers: pseudo-mask mIoU (%) on the COCO training set"
 },
 "results": {
  "big": [["image.jpg", "Image"], ["gt.png", "Ground truth"], ["dhr.png", "DHR (Ours)"]],
  "small": [["tcl.png", "TCL"], ["grounded_sam.png", "Grounded SAM"], ["toco.png", "ToCo"], ["mars.png", "MARS"]],
  "datasets": [
   {"key": "voc", "name": "VOC 2012", "split": "test", "dhr": 79.8, "mars": 77.2, "tags": [["person", [192, 128, 128]], ["sofa", [0, 192, 0]], ["chair", [192, 0, 0]]]},
   {"key": "coco", "name": "COCO 2014", "split": "val", "dhr": 53.9, "mars": 49.4, "tags": [["person", [128, 0, 0]], ["car", [128, 128, 0]], ["motorcycle", [0, 0, 128]], ["umbrella", [64, 192, 0]], ["handbag", [192, 192, 0]]]},
   {"key": "context", "name": "Pascal Context", "split": "val", "dhr": 49.0, "mars": 39.8, "tags": [["dog", [64, 0, 128]], ["floor", [128, 0, 64]], ["food", [128, 128, 64]], ["plate", [192, 0, 192]]]},
   {"key": "ade", "name": "ADE 2016", "split": "val", "dhr": 32.9, "mars": 22.0, "tags": [["table", [255, 6, 82]], ["chair", [204, 70, 3]], ["lamp", [224, 255, 8]], ["ceiling", [120, 120, 80]], ["curtain", [255, 51, 7]], ["sideboard", [255, 112, 0]]]},
   {"key": "stuff", "name": "COCO-Stuff", "split": "val", "dhr": 37.4, "mars": 35.7, "tags": [["person", [128, 0, 0]], ["tennis racket", [128, 128, 192]], ["playingfield", [0, 160, 128]], ["cage", [96, 64, 0]]]}
  ],
  "gap": ["84%", "of the gap between WSS and full supervision is closed on the VOC validation set (RSEPM 74.4, with DHR 79.6, full supervision 80.6)"],
  "message": "Given the same tags, DHR delineates the boundaries between adjacent classes",
  "note": "Paper Fig. 9, Table 2 and Table 4 · validation images of five benchmarks · MARS on Context, ADE and Stuff as reproduced in the paper"
 }
}
