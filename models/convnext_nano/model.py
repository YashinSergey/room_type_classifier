import timm

model = timm.create_model(
    'convnext_nano', 
    pretrained=True, 
    num_classes=20, 
    drop_rate=0.5,       
    drop_path_rate=0.3
).to(DEVICE)