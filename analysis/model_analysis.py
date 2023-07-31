import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from Networks.DAANet import DAA
from config import Config
from datasets.data_utils import DataSetFactory
from tqdm import tqdm

cfg = Config()
###################################
# Load validation data
###################################
factory = DataSetFactory(cfg)
val_loader = DataLoader(factory.testing, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
###################################
# Load Model
###################################
net_info = {
    'da_type': cfg.da_type,
    'feat_dim': cfg.feat_dim,
    'backbone': cfg.backbone,
    'num_classes': cfg.num_classes
}
model = DAA(net_info)
data = torch.load(
    r"C:\Work\repos\Delta_Age_AdaIN\models\megaage_asian_resnet18_100_binary\resnet18_epoch_191_ac_93.75-89.06_best.pth")
state_dict = data['net']
model.load_state_dict(state_dict)
model.cuda()
run_info = {}

run_info['mode'] = 'test'
run_info['accuracy_threshold'] = cfg.accuracy_threshold
if cfg.da_type == 'image_template':
    run_info['template_images'] = self.template_images
    run_info['template_labels'] = self.template_labels

analysis_dict = {}
for img, labels in tqdm(val_loader):
    run_info['labels'] = labels['gt_age'].cuda()
    res = model(img.cuda(), run_info)
    delta = res['age'] - run_info['labels']
    # labels_array=
    for _l in labels['gt_age']:
        index = _l == labels['gt_age']
        int_l = int(_l.item())
        bias = torch.sum(delta[index]).item()
        power = torch.sum(torch.abs(delta[index])).item()
        count = torch.sum(index).item()
        if analysis_dict.get(int_l) is None:
            analysis_dict.update({int_l: [bias,
                                          power,
                                          count]})
        else:
            analysis_dict[int_l][0] += bias
            analysis_dict[int_l][1] += power
            analysis_dict[int_l][2] += count
# print("a")

age_array = np.asarray(list(analysis_dict.keys()))
age_array.sort()


bias_array = [analysis_dict[age][0] / analysis_dict[age][2] for age in age_array]
mse_array = [analysis_dict[age][1] / analysis_dict[age][2] for age in age_array]

mae = sum([analysis_dict[age][1] for age in age_array]) / sum([analysis_dict[age][2] for age in age_array])
print(mae)
plt.subplot(1, 2, 1)
plt.plot(age_array, mse_array)
plt.subplot(1, 2, 2)
plt.plot(age_array, bias_array)
plt.show()
