
# Naive # 
## From @pidnet_finetuning_RGB.ipynb ##
pidnet-fire-smokeNEWDATASETRGB4real-epoch=48-val_fire_smoke_iou=0.708.ckpt

[{'test_loss': 0.008777922950685024,
  'test_lovasz_loss': 0.0,
  'test_focal_loss': 0.0,
  'test_dice_loss': 0.0,
  'test_precision_background': 0.9995138645172119,
  'test_recall_background': 0.9983693361282349,
  'test_f1_background': 0.9989413022994995,
  'test_precision_fire': 0.6216216087341309,
  'test_recall_fire': 0.9426229596138,
  'test_f1_fire': 0.7491856813430786,
  'test_precision_smoke': 0.8445717692375183,
  'test_recall_smoke': 0.9475163817405701,
  'test_f1_smoke': 0.8930873274803162,
  'test_fire_smoke_precision': 0.733096718788147,
  'test_fire_smoke_recall': 0.9450696706771851,
  'test_fire_smoke_f1': 0.821136474609375,
  'test_iou_background': 0.9978874325752258,
  'test_iou_fire': 0.6208241581916809,
  'test_iou_smoke': 0.8068398237228394,
  'test_mean_iou': 0.8085171580314636,
  'test_fire_smoke_iou': 0.7138320207595825}]

# RGCr No tilling, no combine loss (cross-entropy) #
## From @pidnet_finetuning_RGCr_CE_NoTiling.ipynb ##
pidnet-fire-smokeNEWDATASETRGCrAT4REALNOTILLING-epoch=49-val_fire_smoke_iou=0.703.ckpt

[{'test_loss': 0.00271083926782012,
  'test_lovasz_loss': 0.394709050655365,
  'test_focal_loss': 0.00271083926782012,
  'test_dice_loss': 0.505052924156189,
  'test_precision_background': 0.9995095133781433,
  'test_recall_background': 0.9982539415359497,
  'test_f1_background': 0.9988813400268555,
  'test_precision_fire': 0.5750577449798584,
  'test_recall_fire': 0.9764705896377563,
  'test_f1_fire': 0.7238371968269348,
  'test_precision_smoke': 0.8357074856758118,
  'test_recall_smoke': 0.9468985795974731,
  'test_f1_smoke': 0.8878352046012878,
  'test_fire_smoke_precision': 0.7053825855255127,
  'test_fire_smoke_recall': 0.9616845846176147,
  'test_fire_smoke_f1': 0.8058362007141113,
  'test_iou_background': 0.9977882504463196,
  'test_iou_fire': 0.5881032347679138,
  'test_iou_smoke': 0.7998337745666504,
  'test_mean_iou': 0.7952417731285095,
  'test_fire_smoke_iou': 0.6939685344696045}]

# RGB Combined loss no tilling #
## From @pidnet_finetuning_combinedloss.ipynb ##
pidnet-fire-smokeNEWDATASETRGBCombinedloss-epoch=47-val_fire_smoke_iou=0.797.ckpt

[{'test_loss': 0.15201716125011444,
  'test_lovasz_loss': 0.2995207607746124,
  'test_focal_loss': 0.006444149650633335,
  'test_dice_loss': 0.15014341473579407,
  'test_precision_background': 0.9990760087966919,
  'test_recall_background': 0.9988262057304382,
  'test_f1_background': 0.9989510774612427,
  'test_precision_fire': 0.8754098415374756,
  'test_recall_fire': 0.9368420839309692,
  'test_f1_fire': 0.9050847291946411,
  'test_precision_smoke': 0.8762987852096558,
  'test_recall_smoke': 0.899835467338562,
  'test_f1_smoke': 0.8879111409187317,
  'test_fire_smoke_precision': 0.8758543133735657,
  'test_fire_smoke_recall': 0.9183387756347656,
  'test_fire_smoke_f1': 0.8964979648590088,
  'test_iou_background': 0.9978959560394287,
  'test_iou_fire': 0.7993096709251404,
  'test_iou_smoke': 0.7987511157989502,
  'test_mean_iou': 0.8653188943862915,
  'test_fire_smoke_iou': 0.7990304231643677}]

# RGCr Tilling Combined Loss #
## From @pidnet_finetuning_RGCr.ipynb ##
pidnet-fire-smokeNEWDATASETRGCrAT4REAL2-epoch=45-val_fire_smoke_iou=0.756.ckpt

[{'test_precision_background': 0.9989144802093506,
  'test_recall_background': 0.9985754489898682,
  'test_f1_background': 0.9987449049949646,
  'test_precision_fire': 0.8698770999908447,
  'test_recall_fire': 0.9119530916213989,
  'test_f1_fire': 0.890418291091919,
  'test_precision_smoke': 0.8536347150802612,
  'test_recall_smoke': 0.8843548893928528,
  'test_f1_smoke': 0.8687233328819275,
  'test_fire_smoke_precision': 0.861755907535553,
  'test_fire_smoke_recall': 0.8981540203094482,
  'test_fire_smoke_f1': 0.8795708417892456,
  'test_iou_background': 0.9974923729896545,
  'test_iou_fire': 0.8024810552597046,
  'test_iou_smoke': 0.7679139375686646,
  'test_mean_iou': 0.8559624552726746,
  'test_fire_smoke_iou': 0.7851974964141846}]


# RGB Tilling cross-entropy Loss #
pidnet-fire-smokeNEWDATASETRGBTiling14jul-epoch=48-val_fire_smoke_iou=0.735.ckpt

[{'test_precision_background': 0.9994219541549683,
  'test_recall_background': 0.9983981251716614,
  'test_f1_background': 0.9989097714424133,
  'test_precision_fire': 0.6817559599876404,
  'test_recall_fire': 0.9587353467941284,
  'test_f1_fire': 0.7968631982803345,
  'test_precision_smoke': 0.8469641208648682,
  'test_recall_smoke': 0.9381605982780457,
  'test_f1_smoke': 0.8902328610420227,
  'test_fire_smoke_precision': 0.7643600106239319,
  'test_fire_smoke_recall': 0.9484480023384094,
  'test_fire_smoke_f1': 0.8435479998588562,
  'test_iou_background': 0.9978219866752625,
  'test_iou_fire': 0.6623213291168213,
  'test_iou_smoke': 0.8021799325942993,
  'test_mean_iou': 0.8207743763923645,
  'test_fire_smoke_iou': 0.7322506308555603}]
  
