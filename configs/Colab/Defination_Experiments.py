import torch
import yaml
import shutil


def Experiments_GradNorm(Multiple, Clip, File, run):
    if Multiple == 'Yes':
        for Shot in range(0, Clip):
            # Modify and run the mission with GradNorm
            with open(File) as f:
                doc = yaml.safe_load(f)
            doc['data']['GradNorm'] = 'On'
            doc['train']['LossFileName'] = 'eval' + str(Shot + 1) + '_GradNorm'
            with open(File, 'w') as f:
                yaml.safe_dump(doc, f, default_flow_style=False)
            f = open(r'configs/HM/HM_PINO-MBD.yaml')
            HM_config = yaml.load(f)
            Journal = 'checkpoints/' + HM_config['train']['save_dir']
            shutil.copy(File, Journal)
            print('Now running mission {} with GradNorm'.format(HM_config['train']['LossFileName']))
            _ = run(config=HM_config)
            # Modify and run the mission without GradNorm
            with open(File) as f:
                doc = yaml.safe_load(f)
            doc['data']['GradNorm'] = 'Off'
            doc['train']['LossFileName'] = 'eval' + str(Shot + 1) + '_NoGradNorm'
            with open(File, 'w') as f:
                yaml.safe_dump(doc, f, default_flow_style=False)
            f = open(r'configs/HM/HM_PINO-MBD.yaml')
            HM_config = yaml.load(f)
            print('Now running mission {} without GradNorm'.format(HM_config['train']['LossFileName']))
            _ = run(config=HM_config)
    else:
        f = open(r'configs/HM/HM_PINO-MBD.yaml')
        HM_config = yaml.load(f)
        model = run(config=HM_config)

def Experiments_Virtual(Multiple, Clip, File, run):
    if Multiple == 'Yes':
        for Shot in range(0, Clip):
            # Modify and run the mission with GradNorm
            # with open(File) as f:
            #     doc = yaml.safe_load(f)
            # doc['data']['GradNorm'] = 'Off'
            # doc['data']['VirtualSwitch'] = 'On'
            # doc['train']['LossFileName'] = 'eval' + str(Shot + 1) + '_Virtual'
            # with open(File, 'w') as f:
            #     yaml.safe_dump(doc, f, default_flow_style=False)
            # f = open(r'configs/HM/HM_PINO-MBD.yaml')
            # HM_config = yaml.load(f)
            # Journal = 'checkpoints/' + HM_config['train']['save_dir']
            # shutil.copy(File, Journal)
            # print('Now running mission {} with virtual data'.format(HM_config['train']['LossFileName']))
            # _ = run(config=HM_config)
            # Modify and run the mission without GradNorm
            with open(File) as f:
                doc = yaml.safe_load(f)
            doc['data']['GradNorm'] = 'Off'
            doc['data']['VirtualSwitch'] = 'Off'
            doc['train']['LossFileName'] = 'eval' + str(Shot + 1) + '_NoVirtual'
            with open(File, 'w') as f:
                yaml.safe_dump(doc, f, default_flow_style=False)
            f = open(r'configs/HM/HM_PINO-MBD.yaml')
            HM_config = yaml.load(f)
            print('Now running mission {} without virtual data'.format(HM_config['train']['LossFileName']))
            _ = run(config=HM_config)
    else:
        f = open(r'configs/HM/HM_PINO-MBD.yaml')
        HM_config = yaml.load(f)
        model = run(config=HM_config)

def Experiments_GradNorm_BSA(Multiple, Clip, File, run):
    if Multiple == 'Yes':
        for Shot in range(0, Clip):
            # Modify and run the mission with GradNorm
            with open(File) as f:
                doc = yaml.safe_load(f)
            doc['data']['GradNorm'] = 'On'
            doc['train']['LossFileName'] = 'eval' + str(Shot + 1) + '_GradNorm'
            with open(File, 'w') as f:
                yaml.safe_dump(doc, f, default_flow_style=False)
            f = open(r'configs/BSA/BSA_PINO-MBD.yaml')
            HM_config = yaml.load(f)
            Journal = 'checkpoints/' + HM_config['train']['save_dir']
            shutil.copy(File, Journal)
            print('Now running mission {} with GradNorm'.format(HM_config['train']['LossFileName']))
            _ = run(config=HM_config)
            # # Modify and run the mission without GradNorm
            # with open(File) as f:
            #     doc = yaml.safe_load(f)
            # doc['data']['GradNorm'] = 'Off'
            # doc['train']['LossFileName'] = 'eval' + str(Shot + 1) + '_NoGradNorm'
            # with open(File, 'w') as f:
            #     yaml.safe_dump(doc, f, default_flow_style=False)
            # f = open(r'configs/BSA/BSA_PINO-MBD.yaml')
            # HM_config = yaml.load(f)
            # print('Now running mission {} without GradNorm'.format(HM_config['train']['LossFileName']))
            # _ = run(config=HM_config)
    else:
        f = open(r'configs/HM/HM_PINO-MBD.yaml')
        HM_config = yaml.load(f)
        model = run(config=HM_config)

def Experiments_Virtual_BSA(Multiple, Clip, File, run):
    if Multiple == 'Yes':
        for Shot in range(0, Clip):
            # Modify and run the mission with GradNorm
            # with open(File) as f:
            #     doc = yaml.safe_load(f)
            # doc['data']['GradNorm'] = 'Off'
            # doc['data']['VirtualSwitch'] = 'On'
            # doc['train']['LossFileName'] = 'eval' + str(Shot + 1) + '_Virtual'
            # with open(File, 'w') as f:
            #     yaml.safe_dump(doc, f, default_flow_style=False)
            # f = open(r'configs/HM/HM_PINO-MBD.yaml')
            # HM_config = yaml.load(f)
            # Journal = 'checkpoints/' + HM_config['train']['save_dir']
            # shutil.copy(File, Journal)
            # print('Now running mission {} with virtual data'.format(HM_config['train']['LossFileName']))
            # _ = run(config=HM_config)
            # Modify and run the mission without GradNorm
            with open(File) as f:
                doc = yaml.safe_load(f)
            doc['data']['GradNorm'] = 'Off'
            doc['data']['VirtualSwitch'] = 'Off'
            doc['train']['LossFileName'] = 'eval' + str(Shot + 1) + '_NoVirtual'
            with open(File, 'w') as f:
                yaml.safe_dump(doc, f, default_flow_style=False)
            f = open(r'configs/BSA/BSA_PINO-MBD.yaml')
            HM_config = yaml.load(f)
            print('Now running mission {} without virtual data'.format(HM_config['train']['LossFileName']))
            _ = run(config=HM_config)
    else:
        f = open(r'configs/BSA/BSA_PINO-MBD.yaml')
        HM_config = yaml.load(f)
        model = run(config=HM_config)


