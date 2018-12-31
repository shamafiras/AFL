import config
from gui.dc_gan_widget import DcGanWidget
from PySide2.QtWidgets import QApplication


def set_params(widget, opt):
    widget.model.net.alpha1 = widget.alpha1 = opt.a1
    widget.model.net.alpha2 = widget.alpha2 = opt.a2
    widget.model.net.alpha3 = widget.alpha3 = opt.a3
    widget.model.net.alpha4 = widget.alpha4 = opt.a4
    widget.model.net.loop_count = widget.loop_num = opt.loop_count


if __name__ == '__main__':

    manual_seed = 1989  # if not None will set constant generations
    # end of setting configurations
    opt = config.get_configurations()
    # Initialise the application
    app = QApplication([])
    # Call the main widget
    ex = DcGanWidget(app_name=opt.app_name, opt=opt, manual_seed=manual_seed)
    if not opt.eval:
        app.exec_()
    else:
        # set alphas
        set_params(ex, opt)
        # create 50K Samples
        ex.generate_for_eval()