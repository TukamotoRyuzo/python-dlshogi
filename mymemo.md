# python-dlshogi

## 使い方

- python-dlshogiをpythonのモジュールとして使う
  - `pip install --no-cache-dir -e .`
- csa棋譜リストから条件以外の棋譜を取り除いて棋譜を集出する
  - `python utils/filter_csa.py <棋譜リストがあるディレクトリのパス>`
- 抽出した棋譜をシャッフル&訓練データとテストデータに分割する
  - `python utils/make_kifu_list.py <棋譜リストがあるディレクトリのパス> <保存するファイル名のプレフィクス>`
  - 実行するとカレントディレクトリに、以下の2つのファイルができあがる
    - <保存するファイル名のプレフィクス>_train.txt
    - <保存するファイル名のプレフィクス>_test.txt
- 学習実行
  - `python train_policy.py <トレーニング用棋譜リスト> <テスト用棋譜リスト> --eval_interval <テストデータを評価する間隔>`

## Google colaboratory上で実行するための環境構築

- はやく[`Google colaboratory`](https://colab.research.google.com/)にアクセスするんだ
- `GPU`をアサインしよう(これ忘れてるとGPU使えないぞ)
  - 画面上部のメニュー ランタイム -> ランタイムのタイプを変更 -> ノートブックの設定 を開く
  - ハードウェアアクセラレータに GPU を選択し、 保存 する

### Deep Leaning用のライブラリを入れよう

- `Keras`は以下のコマンドを実行（しなくてもいいかも）[[1]](https://qiita.com/tomo_makes/items/f70fe48c428d3a61e131) [[2]](https://qiita.com/stakemura/items/1761be70a06fa8ee853f#chainer)
  - `!pip install keras`
- `chainer`ならこう

``` shell
!apt-get install -y -qq libcusparse8.0 libnvrtc8.0 libnvtoolsext1
!ln -snf /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so.8.0 /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so
!pip install cupy-cuda80==4.0.0b4 
!pip install chainer==4.0.0b4
```

### Google Drive をマウントしよう

- Colaboratoryは大容量データのアップロードが苦手なので、そういうのはGoogle Driveに入れとこう

- 以下のコマンドを実行する [[3]](https://qiita.com/uni-3/items/201aaa2708260cc790b8)

``` python
from google.colab import drive
drive.mount('/content/gdrive')
```

- gdriveのMy Drive以下にdriveのルートディレクトリがマウントされている
- 嘘だと思うなら以下のコマンドを実行
  - `!ls ./gdrive/'My Drive'`

### `github`からリポジトリをクローンしよう

- `!git clone https://github.com/TukamotoRyuzo/python-dlshogi.git`

### 棋譜を用意しよう

- Google Driveに棋譜入れとこう
- zip圧縮されてない状態で入れておこう
- zip圧縮されていたらunzipしよう
  - `!unzip ./gdrive/'My Drive'/wdoor2016.zip`
  - とても時間かかるよ

### python-dlshogiを使ってみよう

- ディレクトリ移動
  - `cd python-dlshogi`
- 必要なもの入れとく
  - `!pip install --no-cache-dir -e .`
  - `!pip install python-shogi`
- 棋譜のフィルタリング
  - `!python utils/filter_csa.py "../gdrive/My Drive/wdoor2016/2016"`
  - __これcolaboratory上だと時間かかりすぎるから諦めた方が良い__
- 棋譜リスト作ろう
  - `!python utils/make_kifu_list.py "../gdrive/My Drive/wdoor2016/2016" kifulist`
- 学習を回そう
  - 少なめの学習データで実行
    - `!python train_policy.py kifulist_train_1000.txt kifulist_test_100.txt --eval_interval 100`
  - 普通に実行
    - `!python train_policy.py kifulist_train.txt kifulist_test.txt --eval_interval 10000`
## ディレクトリ構成

- bat
  - いろんな探索方法で遊ばせてみるためのコマンド
- model
  - モデルの重みファイル(policyとvalue)
- pydlshogi
  - read_kifu.py
    - 棋譜の読み込み
  - features.py
    - 局面 -> 入力特徴
    - 指し手 -> 出力ラベル
  - common.py
    - 定数定義とか盤面回転処理とか
  - network
    - policy.py
      - 方策ネットワーク(policy network)の実装
  - player
    - いろんな探索方法で探索してみるプレイヤー実装
  - uct
    - uct探索実装
  - usi
    - playerをusiエンジン化する実装

- utils
  - 教師のお掃除とかログ出力とかもろもろ
  - filter_csa.py
    - csa棋譜リストから条件以外の棋譜を取り除いて棋譜を集出する
  - make_kifu_list.py
    - 抽出した棋譜をシャッフル&訓練データとテストデータに分割する
- train_policy.py
  - 方策ネットワークを学習する
