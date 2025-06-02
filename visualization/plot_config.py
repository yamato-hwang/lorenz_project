# visualization/plot_config.py
import matplotlib as mpl

# 한글 폰트 (환경에 맞게 하나만 활성화)
mpl.rc('font', family='AppleGothic')   # macOS
# mpl.rc('font', family='Malgun Gothic')  # Windows
# mpl.rc('font', family='NanumGothic')   # Linux

# 마이너스 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False
