import yaml
import torch
import shutil
import xlrd
import pandas as pd


def read_HPA_table(file):
    tables = []
    wb = xlrd.open_workbook(filename=file)
    table = wb.sheets()[0]
    for rown in range(table.nrows):
        if rown == 0:
            continue
        elme = []
        for i in range(table.row_len(rown)):
            elme.append(table.cell_value(rown, i))
        tables.append(elme)
    return tables


def yaml_update(params, yaml_file):
    # Information in params: Case Name; GKN_width; GKN_Depth; FCN_Width; FCN_Depth
    # Update the config file
    with open(yaml_file) as f:
        doc = yaml.safe_load(f)
    doc['log']['Case'] = params[0]
    doc['train']['LossFileName'] = params[0]
    doc['train']['save_name'] = params[0] + '.pt'
    doc['model']['width'] = int(params[1])
    doc['model']['depth'] = int(params[2])
    doc['model']['fc_dim'] = int(params[3])
    doc['model']['fc_dep'] = int(params[4])
    # Safe insurance
    doc['data']['GradNorm'] = 'Off'
    doc['data']['NoData'] = 'Off'
    doc['data']['VirtualSwitch'] = 'Off'
    # Update the loss function type
    loss_type = params[5]

    if loss_type == 'dde':
        # Remember to change this for other cases!!!!!!!!!!!!!
        doc['data']['DiffLossSwitch'] = 'On'
        doc['data']['Boundary'] = 'Off'
        doc['data']['VirtualSwitch'] = 'Off'
        print('This case is using dde loss.')
    elif loss_type == 'eq':
        doc['data']['DiffLossSwitch'] = 'Off'
        doc['data']['Boundary'] = 'Off'
        doc['data']['VirtualSwitch'] = 'Off'
        print('This case is using eq loss.')
    else:
        print('Warning! Not receiving loss function type correctly.')
    # Complete the update
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)
    with open(yaml_file) as f:
        doc = yaml.safe_load(f)
    # Copy the yaml file
    Journal = 'checkpoints/' + doc['train']['save_dir'] + '/' + doc['log']['Case'] + '.yaml'
    shutil.copy(yaml_file, Journal)
    print('Yaml file has been overwritten and documented.')
    print('You have: Case Name = {} ; GKN_width = {} ; GKN_Depth = {} ; FCN_Width = {} ; FCN_Depth = {} '.format(doc['log']['Case'], doc['model']['width'], doc['model']['depth'], doc['model']['fc_dim'], doc['model']['fc_dep']))


def yaml_update_BSA(params, yaml_file):
    # Information in params: Case Name; GKN_width; GKN_Depth; FCN_Width; FCN_Depth
    # Update the config file
    with open(yaml_file) as f:
        doc = yaml.safe_load(f)
    doc['log']['Case'] = params[0]
    doc['train']['LossFileName'] = params[0]
    doc['train']['save_name'] = params[0] + '.pt'
    doc['model']['width'] = int(params[1])
    doc['model']['depth'] = int(params[2])
    doc['model']['fc_dim'] = int(params[3])
    doc['model']['fc_dep'] = int(params[4])
    # Safe insurance
    doc['data']['GradNorm'] = 'Off'
    doc['data']['NoData'] = 'Off'
    doc['data']['VirtualSwitch'] = 'Off'
    # Update the loss function type
    loss_type = params[5]

    if loss_type == 'dde':
        # Remember to change this for other cases!!!!!!!!!!!!!
        doc['data']['DiffLossSwitch'] = 'On'
        doc['data']['Boundary'] = 'Off'
        doc['data']['VirtualSwitch'] = 'On'
        print('This case is using dde loss.')
    elif loss_type == 'eq':
        doc['data']['DiffLossSwitch'] = 'On'
        doc['data']['Boundary'] = 'On'
        doc['data']['VirtualSwitch'] = 'On'
        print('This case is using eq loss.')
    else:
        print('Warning! Not receiving loss function type correctly.')
    # Complete the update
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)
    with open(yaml_file) as f:
        doc = yaml.safe_load(f)
    # Copy the yaml file
    Journal = 'checkpoints/' + doc['train']['save_dir'] + '/' + doc['log']['Case'] + '.yaml'
    shutil.copy(yaml_file, Journal)
    print('Yaml file has been overwritten and documented.')
    print('You have: Case Name = {} ; GKN_width = {} ; GKN_Depth = {} ; FCN_Width = {} ; FCN_Depth = {} '.format(doc['log']['Case'], doc['model']['width'], doc['model']['depth'], doc['model']['fc_dim'], doc['model']['fc_dep']))


def yaml_update_VTCD(params, yaml_file):
    # Information in params: Case Name; GKN_width; GKN_Depth; FCN_Width; FCN_Depth
    # Update the config file
    with open(yaml_file) as f:
        doc = yaml.safe_load(f)
    doc['log']['Case'] = params[0]
    doc['train']['LossFileName'] = params[0]
    doc['train']['save_name'] = params[0] + '.pt'
    doc['model']['width'] = int(params[1])
    doc['model']['depth'] = int(params[2])
    doc['model']['fc_dim'] = int(params[3])
    doc['model']['fc_dep'] = int(params[4])
    # Safe insurance
    doc['data']['GradNorm'] = 'Off'
    doc['data']['NoData'] = 'Off'
    doc['data']['VirtualSwitch'] = 'Off'
    # Update the loss function type
    loss_type = params[5]

    if loss_type == 'dde':
        # Remember to change this for other cases!!!!!!!!!!!!!
        doc['data']['DiffLossSwitch'] = 'On'
        doc['data']['Boundary'] = 'Off'
        doc['data']['VirtualSwitch'] = 'Off'
        print('This case is using dde loss.')
    elif loss_type == 'eq':
        doc['data']['DiffLossSwitch'] = 'On'
        doc['data']['Boundary'] = 'On'
        doc['data']['VirtualSwitch'] = 'Off'
        print('This case is using eq loss.')
    else:
        print('Warning! Not receiving loss function type correctly.')
    # Complete the update
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)
    with open(yaml_file) as f:
        doc = yaml.safe_load(f)
    # Copy the yaml file
    Journal = 'checkpoints/' + doc['train']['save_dir'] + '/' + doc['log']['Case'] + '.yaml'
    shutil.copy(yaml_file, Journal)
    print('Yaml file has been overwritten and documented.')
    print('You have: Case Name = {} ; GKN_width = {} ; GKN_Depth = {} ; FCN_Width = {} ; FCN_Depth = {} '.format(doc['log']['Case'], doc['model']['width'], doc['model']['depth'], doc['model']['fc_dim'], doc['model']['fc_dep']))


def yaml_update_HM(params, yaml_file):
    # Information in params: Case Name; GKN_width; GKN_Depth; FCN_Width; FCN_Depth
    # Update the config file
    with open(yaml_file) as f:
        doc = yaml.safe_load(f)
    doc['log']['Case'] = params[0]
    doc['train']['LossFileName'] = params[0]
    doc['train']['save_name'] = params[0] + '.pt'
    doc['model']['width'] = int(params[1])
    doc['model']['depth'] = int(params[2])
    doc['model']['fc_dim'] = int(params[3])
    doc['model']['fc_dep'] = int(params[4])
    # Safe insurance
    doc['data']['GradNorm'] = 'Off'
    doc['data']['NoData'] = 'Off'
    doc['data']['VirtualSwitch'] = 'Off'
    # Update the loss function type
    loss_type = params[5]

    if loss_type == 'dde':
        # Remember to change this for other cases!!!!!!!!!!!!!
        doc['data']['DiffLossSwitch'] = 'On'
        doc['data']['Boundary'] = 'Off'
        doc['data']['VirtualSwitch'] = 'Off'
        print('This case is using dde loss.')
    elif loss_type == 'eq':
        doc['data']['DiffLossSwitch'] = 'Off'
        doc['data']['Boundary'] = 'Off'
        doc['data']['VirtualSwitch'] = 'Off'
        print('This case is using eq loss.')
    else:
        print('Warning! Not receiving loss function type correctly.')
    # Complete the update
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)
    with open(yaml_file) as f:
        doc = yaml.safe_load(f)
    # Copy the yaml file
    Journal = 'checkpoints/' + doc['train']['save_dir'] + '/' + doc['log']['Case'] + '.yaml'
    shutil.copy(yaml_file, Journal)
    print('Yaml file has been overwritten and documented.')
    print('You have: Case Name = {} ; GKN_width = {} ; GKN_Depth = {} ; FCN_Width = {} ; FCN_Depth = {} '.format(doc['log']['Case'], doc['model']['width'], doc['model']['depth'], doc['model']['fc_dim'], doc['model']['fc_dep']))


def yaml_BOupdate(a, b, c, d, loss_type, yaml_file):
    with open(yaml_file) as f:
        doc = yaml.safe_load(f)
    doc['model']['width'] = int(a)
    doc['model']['depth'] = int(b)
    doc['model']['fc_dim'] = int(c)
    doc['model']['fc_dep'] = int(d)
    # Safe insurance
    doc['data']['GradNorm'] = 'Off'
    doc['data']['NoData'] = 'Off'
    doc['data']['VirtualSwitch'] = 'Off'

    if loss_type == 'dde':
        # Remember to change this for other cases!!!!!!!!!!!!!
        doc['data']['DiffLossSwitch'] = 'On'
        doc['data']['Boundary'] = 'Off'
        doc['train']['LossFileName'] = 'BO_dde'
        print('This case is using dde loss.')
    elif loss_type == 'eq':
        doc['data']['DiffLossSwitch'] = 'On'
        doc['data']['Boundary'] = 'On'
        doc['train']['LossFileName'] = 'BO_eq'
        print('This case is using eq loss.')
    else:
        print('Warning! Not receiving loss function type correctly.')
    # Complete the update
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)
    with open(yaml_file) as f:
        doc = yaml.safe_load(f)
    print('You have: Case Name = {} ; GKN_width = {} ; GKN_Depth = {} ; FCN_Width = {} ; FCN_Depth = {} '.format(doc['log']['Case'], doc['model']['width'], doc['model']['depth'], doc['model']['fc_dim'], doc['model']['fc_dep']))


def yaml_general_update(excel_file, index, yaml_file):
    df = pd.read_excel(excel_file)

    variables = ['Index', 'CaseName', 'GKN-Width', 'GKN-Depth', 'FCN-Width', 'FCN-Depth', 'GradNorm',
                 'Diff', 'Boundary', 'Virtual', 'fv_weight', 'diff_weight', 'lr', 'epochs', 'StepGap', 'StepRatio', 'Directory', 'Note']
    case_data = []
    for _, row in df.iterrows():
        params = {var: row[var] for var in variables}
        case_data.append(params)
    # Pick the right case information (using the index)
    case_data = [data for data in case_data if data['Index'] == index][0]
    print(case_data['CaseName'])
    function_yaml_update(case_data, yaml_file)
    StepGap = case_data['StepGap']
    StepRatio = case_data['StepRatio']
    return StepGap, StepRatio


def function_yaml_update(case_data, yaml_file):
    with open(yaml_file) as f:
        doc = yaml.safe_load(f)
    doc['train']['LossFileName'] = case_data['CaseName']
    doc['train']['save_name'] = case_data['CaseName']
    doc['train']['save_dir'] = case_data['Directory']
    doc['model']['width'] = case_data['GKN-Width']
    doc['model']['depth'] = case_data['GKN-Depth']
    doc['model']['fc_dim'] = case_data['FCN-Width']
    doc['model']['fc_dep'] = case_data['FCN-Depth']
    doc['data']['GradNorm'] = case_data['GradNorm']
    doc['data']['DiffLossSwitch'] = case_data['Diff']
    doc['data']['Boundary'] = case_data['Boundary']
    doc['data']['VirtualSwitch'] = case_data['Virtual']
    doc['train']['fv_loss'] = case_data['fv_weight']
    doc['train']['diff_loss'] = case_data['diff_weight']
    doc['train']['base_lr'] = case_data['lr']
    doc['train']['epochs'] = case_data['epochs']
    doc['train']['save_dir'] = case_data['Directory']
    if case_data['Note'] == 'PINO':
        print('Oops! You are using naked PINO weights!')
        doc['weights_datapath'] = 'data/Project_BSA/Weights_PINO_150.mat'
        doc['weights_datapath_virtual'] = 'data/Project_BSA/Weights_PINO_Virtual.mat'
    else:
        print('Yes! You are using EN weights!')
        doc['weights_datapath'] = 'data/Project_BSA/Weights_Medium_150.mat'
        doc['weights_datapath_virtual'] = 'data/Project_BSA/Weights_Virtual.mat'

    # Complete the update
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)
    with open(yaml_file) as f:
        doc = yaml.safe_load(f)
    print('*** You have: Case Name = {} ;\n*** GKN_width = {} ; GKN_Depth = {} ; FCN_Width = {} ; FCN_Depth = {},\n*** GradNorm={},\n*** VirtualSwitch={}, fv_weight={}, BoundarySwitch={}, diff_weight={}\n*** LR={}, StepGap={}, StepRatio={}, Directory={}'.format(doc['train']['LossFileName'], doc['model']['width'], doc['model']['depth'], doc['model']['fc_dim'], doc['model']['fc_dep'], doc['data']['GradNorm'], doc['data']['VirtualSwitch'], doc['train']['fv_loss'], doc['data']['Boundary'], doc['train']['diff_loss'], doc['train']['base_lr'], case_data['StepGap'], case_data['StepRatio'], doc['train']['save_dir']))
