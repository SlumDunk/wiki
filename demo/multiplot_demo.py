import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib import animation
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def hello_world():
    x = np.linspace(-1, 1, 50)
    # y = 2 * x + 1
    y = x ** 2
    plt.plot(x, y)
    plt.show()


pass


# figure 独立的图
def graph_framework():
    x = np.linspace(-1, 1, 50)
    y1 = 2 * x + 1
    y2 = x ** 2
    plt.figure()
    plt.plot(x, y1)

    plt.figure(num=3, figsize=(8, 5))
    # 加图例
    line1, = plt.plot(x, y2, label='up')
    line2, = plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='down')

    # 设置坐标轴取值范围
    plt.xlim((-1, 2))
    plt.ylim((-2, 3))

    plt.xlabel('I am x')
    plt.ylabel('I am y')

    new_ticks = np.linspace(-1, 2, 5)
    print(new_ticks)
    plt.xticks(new_ticks)
    plt.yticks([-2, -1.8, -1, 1.22, 3, ], [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])

    # 移动坐标轴位置
    # gca = 'get current axis' 代码没提示
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))  # outward, axes
    ax.spines['left'].set_position(('data', 0))

    # 加图例
    plt.legend(handles=[line1, line2], labels=['aaa', 'bbb'], loc='best')

    # 加标注
    x0 = 1
    y0 = 2 * x0 + 1
    plt.scatter(x0, y0, s=50, color='b')
    plt.plot([x0, x0], [y0, 0], 'k--', lw=2.5)
    plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30), textcoords='offset points',
                 fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    plt.text(-3.7, 3, r'$this\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$', fontdict={'size': 16, 'color': 'red'})

    # ticks能见度
    plt.figure()
    plt.plot(x, y1, linwidth=10)
    plt.ylim((-2, 2))
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))  # outward, axes
    ax.spines['left'].set_position(('data', 0))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))

    plt.show()


pass


def scatter():
    """

    """
    n = 1024
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    T = np.arctan2(Y, X)  # for color value
    plt.scatter(X, Y, s=75, c=T, alpha=0.5)
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    # 隐藏ticks
    plt.xticks(())
    plt.yticks(())
    plt.show()
    pass


def barchart():
    """

    """
    n = 12
    X = np.arange(n)
    Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    plt.bar(X, Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
    for x, y in zip(X, Y1):
        plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

    for x, y in zip(X, Y2):
        plt.text(x + 0.4, -y - 0.05, '-%.2f' % y, ha='center', va='top')
    # 隐藏ticks
    plt.xticks(())
    plt.yticks(())
    plt.show()
    pass


def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


def contour():
    """

    """
    n = 256
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)

    # 10个圈
    plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.cm.hot)
    C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
    plt.clabel(C, inline=True, fontsize=10)
    plt.show()
    pass


def heatmap():
    """

    """
    a = np.arange(9).reshape(3, 3)
    plt.imshow(a, interpolation='nearest', cmap='bone', origin='upper')
    plt.colorbar(shrink=0.9)

    # 隐藏ticks
    plt.xticks(())
    plt.yticks(())
    plt.show()
    pass


def img3d():
    """

    """
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.contourf(X, Y, Z, offset=-2, cmap='rainbow')
    ax.set_zlim(-2, 2)
    plt.show()


pass


def multi_subplots():
    """

    """
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot([0, 1][0, 1])
    plt.subplot(2, 2, 2)
    plt.plot([0, 1][0, 2])
    plt.subplot(223)
    plt.plot([0, 1][0, 3])
    plt.subplot(224)
    plt.plot([0, 1][0, 4])

    # subplot2grid
    plt.figure()
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    ax1.plot([1, 2], [1, 2])
    ax1.set_title('ax1_title')

    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax5 = plt.subplot2grid((3, 3), (2, 1))

    # gridspec
    plt.figure()
    gs = gridspec.GridSpec(3, 3)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :2])
    ax3 = plt.subplot(gs[1:, 2])
    ax4 = plt.subplot(gs[-1, 0])
    ax5 = plt.subplot(gs[-1, -2])

    # easy to define structure
    f, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, sharex=True, sharey=True)
    ax11.scatter([1, 2], [1, 2])
    plt.show()


pass


def img_in_img():
    fig = plt.figure()
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 3, 4, 2, 5, 8, 6]

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.plot(x, y, 'r')
    ax1.set_title('title')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(y, x, 'b')
    ax2.set_title('title inside 1')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.axes([0.6, 0.2, 0.25, 0.25])
    plt.plot(y[::-1], x, 'g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('title inside 2')
    plt.show()


pass


def main_axis():
    x = np.arange(0, 10, 0.1)
    y1 = 0.05 * x ** 2
    y2 = -1 * y1
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y1, 'g-')
    ax2.plot(x, y2, 'b-')

    ax1.set_xlabel('X data')
    ax1.set_ylabel('Y1', color='g')
    ax2.set_ylabel('Y2', color='b')
    plt.show()


pass

if __name__ == '__main__':
    # hello_world()
    # graph_framework()
    # scatter()
    # contour()
    heatmap()