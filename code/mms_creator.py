import code.mms_data.base
import code.mms_data.mms
import yaml

def create_mms():
    import sys
    import os
    from mako.template import Template
    dirname = os.path.dirname(__file__)
    mako_file = os.path.join(dirname, 'mms_core.mako')
    mytemplate = Template(filename=mako_file)
    yaml_file = os.path.join(dirname, 'mms_data', 'mms.yaml')
    if os.path.exists(yaml_file):
        fp = open(yaml_file, 'r')
    else:
        import sys
        print(yaml_file, 'not found')
        sys.exit()
    data = yaml.load(fp, Loader=yaml.FullLoader)['py']
    for key in data:
        code = mytemplate.render(formula=data[key])
        dirname = os.path.dirname(__file__)
        outfile = os.path.join(dirname , key+'.py')
        print('creating', outfile)
        fileptr = open(outfile, 'w')
        fileptr.writelines(code)
        fileptr.close() 