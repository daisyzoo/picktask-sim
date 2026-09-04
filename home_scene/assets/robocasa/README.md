# RoboCasa 资产目录（默认为空）
#
# 接入方式见仓库根 README 与:
#   python scripts/fetch_robocasa_assets.py --help
#
# 推荐轻量下载（需先 pip install robocasa）:
#   python scripts/fetch_robocasa_assets.py --download --type fixtures_lw objs_lw tex
#   python scripts/fetch_robocasa_assets.py --link --scan
#
# 或:
#   export HOME_SCENE_ROBOCASA_ROOT=/path/to/robocasa/models/assets
#   python scripts/fetch_robocasa_assets.py --from-env --scan
#
# 下载后 XML 会出现在本目录；scene_composer 仍默认用程序化 stub 柜，
# 避免 G1 场景被 RoboCasa/Franka 坐标系与 mesh 路径强绑定。
# registry 会自动扫描并登记，供后续替换 fixture 使用。
