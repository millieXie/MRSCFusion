
class args():
	# training args
	epochs = 2
	batch_size = 4

	dataset_ir = "/hy-tmp/change_data_else_1"
	dataset_vis = "/hy-tmp/change_data_else_1"

	dataset = 'medical'

	HEIGHT = 256
	WIDTH = 256

	save_fusion_model = "models/train/fusionnet"
	save_loss_dir = 'models/train/loss_fusionnet'

	image_size = 256
	cuda = 1
	seed = 42

	lr = 1e-4
	log_interval = 10
	resume_fusion_model = None
	# nest net model
	resume_nestfuse = 'models/model/nestfuse_gray_1e2.model'
	resume_vit = './imagenet21k+imagenet2012_ViT-L_16.pth'
	fusion_model = './models/rfn_twostage/'

	mode = "fusion_axial"




