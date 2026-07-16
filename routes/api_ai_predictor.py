# core/routes/api_ai_predictor.py
"""
AI 智慧预测引擎 - 后端 API（完整版）
包含：模型训练、概率校准、动态策略推荐、AI 机器人控制、价格目标预测
"""

import os
import sys
import pickle
import json
import threading
import logging
import time
from datetime import datetime
from flask import Blueprint, request, jsonify, Response, stream_with_context
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.ai.feature_factory import FeatureFactory
from core.ai.label_generator import LabelGenerator
from core.ai.dynamic_strategy_recommender import get_recommended_strategy

api_ai_predictor = Blueprint('api_ai_predictor', __name__, url_prefix='/api/ai-predictor')
logger = logging.getLogger(__name__)

# 模型存储目录
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ai_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 训练任务状态
_training_status = {'running': False, 'progress': 0, 'message': '', 'done': False}

# AI 机器人实例
_ai_bot_instance = None
_bot_lock = threading.Lock()


# ============ 模型管理 ============

@api_ai_predictor.route('/models', methods=['GET'])
def get_models():
    """获取所有已训练的模型列表"""
    try:
        models = []
        for f in os.listdir(MODEL_DIR):
            if f.endswith('.pkl') and 'scaler' not in f and 'feature_cols' not in f and 'calibrators' not in f:
                parts = f.replace('.pkl', '').split('_')
                is_price_target = 'price_target' in f.lower()
                models.append({
                    'name': f,
                    'symbol': parts[0] if len(parts) > 0 else 'Unknown',
                    'timeframe': parts[1] if len(parts) > 1 else 'H1',
                    'epochs': parts[2] if len(parts) > 2 else '?',
                    'type': 'price_target' if is_price_target else 'state',
                    'created_at': datetime.fromtimestamp(os.path.getctime(os.path.join(MODEL_DIR, f))).isoformat()
                })
        return jsonify({
            'success': True,
            'data': sorted(models, key=lambda x: x['created_at'], reverse=True)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_ai_predictor.route('/models/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """删除指定模型"""
    try:
        file_path = os.path.join(MODEL_DIR, model_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': f'模型 {model_name} 已删除'})
        return jsonify({'success': False, 'error': '模型不存在'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============ 模型训练 ============

@api_ai_predictor.route('/train', methods=['POST'])
def train_model():
    """启动状态模型训练（异步）"""
    global _training_status
    
    data = request.get_json()
    symbol = data.get('symbol', 'BTCUSDm')
    timeframe = data.get('timeframe', 'H1')
    epochs = int(data.get('epochs', 60))
    
    _training_status = {
        'running': True,
        'progress': 0,
        'message': '准备数据...',
        'done': False,
        'error': None,
        'type': 'state'
    }
    
    def train_thread():
        global _training_status
        try:
            # 1. 加载数据
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'backtest_data')
            data_file = os.path.join(data_dir, f'{symbol}_{timeframe}_data.csv')
            
            if not os.path.exists(data_file):
                _training_status = {
                    'running': False, 'done': True, 
                    'error': f'数据文件不存在: {data_file}',
                    'progress': 0, 'message': '数据文件不存在',
                    'type': 'state'
                }
                return
            
            df = pd.read_csv(data_file, parse_dates=['time'])
            _training_status['progress'] = 20
            _training_status['message'] = f'加载了 {len(df)} 条数据'
            
            # 2. 计算特征
            df_feat = FeatureFactory.compute_features(df)
            _training_status['progress'] = 40
            _training_status['message'] = '计算特征完成'
            
            # 3. 生成标签
            labels = LabelGenerator.generate_labels(df_feat, forward_bars=10)
            df_feat['label'] = labels
            df_feat = df_feat.dropna(subset=['label'])
            _training_status['progress'] = 60
            _training_status['message'] = f'生成 {len(df_feat)} 个样本'
            
            # 4. 特征和标签分离
            exclude = ['label', 'time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']
            X = df_feat.drop(columns=[c for c in exclude if c in df_feat.columns])
            y = df_feat['label'].astype(int)
            
            # 5. 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            feature_cols = list(X.columns)
            
            _training_status['progress'] = 70
            _training_status['message'] = f'训练 XGBoost 模型 (Epochs={epochs})...'
            
            # 6. 训练（带类别权重）
            from sklearn.utils.class_weight import compute_class_weight
            from collections import Counter
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.array([class_weights[int(label)] for label in y])
            
            model = XGBClassifier(
                n_estimators=epochs,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            model.fit(X_scaled, y, sample_weight=sample_weights)
            
            # 7. 概率校准
            _training_status['progress'] = 85
            _training_status['message'] = '概率校准...'
            
            from sklearn.isotonic import IsotonicRegression
            from sklearn.model_selection import cross_val_predict
            n_classes = len(np.unique(y))
            calibrators = []
            for i in range(n_classes):
                try:
                    cv_proba = cross_val_predict(
                        model, X_scaled, y, 
                        cv=3, method='predict_proba', n_jobs=-1
                    )[:, i]
                    iso_reg = IsotonicRegression(out_of_bounds='clip')
                    iso_reg.fit(cv_proba, (y == i).astype(int))
                    calibrators.append(iso_reg)
                except Exception:
                    calibrators.append(None)
            model.calibrators = calibrators
            
            _training_status['progress'] = 90
            _training_status['message'] = '保存模型...'
            
            # 8. 保存
            model_name = f"{symbol}_{timeframe}_{epochs}.pkl"
            scaler_name = f"{symbol}_{timeframe}_scaler.pkl"
            feature_name = f"{symbol}_{timeframe}_feature_cols.pkl"
            calibrator_name = f"{symbol}_{timeframe}_calibrators.pkl"
            
            with open(os.path.join(MODEL_DIR, model_name), 'wb') as f:
                pickle.dump(model, f)
            with open(os.path.join(MODEL_DIR, scaler_name), 'wb') as f:
                pickle.dump(scaler, f)
            with open(os.path.join(MODEL_DIR, feature_name), 'wb') as f:
                pickle.dump(feature_cols, f)
            with open(os.path.join(MODEL_DIR, calibrator_name), 'wb') as f:
                pickle.dump(calibrators, f)
            
            _training_status['progress'] = 100
            _training_status['message'] = f'✅ 模型训练完成: {model_name}'
            _training_status['done'] = True
            _training_status['running'] = False
            _training_status['model_name'] = model_name
            _training_status['type'] = 'state'
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            _training_status = {
                'running': False, 'done': True, 'error': str(e),
                'progress': 0, 'message': f'训练失败: {e}',
                'type': 'state'
            }
    
    thread = threading.Thread(target=train_thread, daemon=True)
    thread.start()
    return jsonify({'success': True, 'message': '训练已开始'})


@api_ai_predictor.route('/train/status', methods=['GET'])
def get_train_status():
    """获取训练状态"""
    return jsonify({
        'success': True,
        'data': _training_status
    })


@api_ai_predictor.route('/train/cancel', methods=['POST'])
def cancel_training():
    """取消训练"""
    global _training_status
    if _training_status['running']:
        _training_status['running'] = False
        _training_status['done'] = True
        _training_status['message'] = '训练已取消'
        return jsonify({'success': True, 'message': '已发送取消信号'})
    return jsonify({'success': False, 'error': '没有正在运行的任务'})


# ============ 价格目标模型训练 ============

@api_ai_predictor.route('/train/price-target', methods=['POST'])
def train_price_target():
    """训练价格目标预测模型"""
    global _training_status
    
    data = request.get_json()
    symbol = data.get('symbol', 'XAUUSDm')
    timeframe = data.get('timeframe', 'M5')
    forward_bars = int(data.get('forward_bars', 20))
    
    _training_status = {
        'running': True,
        'progress': 0,
        'message': '准备训练价格目标模型...',
        'done': False,
        'error': None,
        'type': 'price_target'
    }
    
    def train_thread():
        global _training_status
        try:
            from core.ai.price_target_predictor import PriceTargetPredictor
            
            # 1. 加载数据
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'backtest_data')
            data_file = os.path.join(data_dir, f'{symbol}_{timeframe}_data.csv')
            
            if not os.path.exists(data_file):
                _training_status = {
                    'running': False, 'done': True, 
                    'error': f'数据文件不存在: {data_file}',
                    'progress': 0, 'message': '数据文件不存在',
                    'type': 'price_target'
                }
                return
            
            df = pd.read_csv(data_file, parse_dates=['time'])
            _training_status['progress'] = 10
            _training_status['message'] = f'加载了 {len(df)} 条数据'
            
            # 2. 训练
            _training_status['progress'] = 20
            _training_status['message'] = '计算特征并训练价格目标模型...'
            
            predictor = PriceTargetPredictor()
            predictor.train(df, forward_bars=forward_bars)
            
            _training_status['progress'] = 80
            _training_status['message'] = '保存模型...'
            
            # 3. 保存
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ai_models')
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f'{symbol}_{timeframe}_price_target.pkl')
            predictor.save(model_path)
            
            _training_status['progress'] = 100
            _training_status['message'] = f'✅ 价格目标模型训练完成: {symbol}_{timeframe}_price_target.pkl'
            _training_status['done'] = True
            _training_status['running'] = False
            _training_status['model_name'] = f'{symbol}_{timeframe}_price_target.pkl'
            _training_status['type'] = 'price_target'
            
        except Exception as e:
            logger.error(f"价格目标训练失败: {e}")
            _training_status = {
                'running': False, 'done': True, 'error': str(e),
                'progress': 0, 'message': f'训练失败: {e}',
                'type': 'price_target'
            }
    
    thread = threading.Thread(target=train_thread, daemon=True)
    thread.start()
    
    return jsonify({'success': True, 'message': '价格目标训练已开始'})


# ============ AI 预测（自动识别模型类型） ============

@api_ai_predictor.route('/predict', methods=['POST'])
def predict():
    """AI 预测 - 自动识别状态模型或价格目标模型"""
    data = request.get_json()
    symbol = data.get('symbol', 'BTCUSDm')
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({'success': False, 'error': '请选择模型'}), 400
    
    # 检测模型类型
    is_price_target = 'price_target' in model_name.lower()
    
    if is_price_target:
        return _predict_price_target(symbol, model_name)
    else:
        return _predict_state(symbol, model_name)


def _predict_price_target(symbol: str, model_name: str):
    """价格目标预测"""
    try:
        from core.ai.price_target_predictor import PriceTargetPredictor
        
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': '模型文件不存在'}), 404
        
        # 加载模型
        predictor = PriceTargetPredictor()
        predictor.load(model_path)
        
        # 获取市场数据（使用 M5）
        from core.utils.mt5 import get_rates_mt5
        import MetaTrader5 as mt5
        df = get_rates_mt5(symbol, mt5.TIMEFRAME_M5, 150)
        
        if df is None or df.empty:
            # 如果 M5 获取失败，尝试 H1
            df = get_rates_mt5(symbol, mt5.TIMEFRAME_H1, 150)
            if df is None or df.empty:
                return jsonify({'success': False, 'error': '无法获取市场数据'}), 400
        
        # 预测
        target_info = predictor.predict(df)
        
        if target_info.get('direction') == 'UNKNOWN':
            return jsonify({
                'success': True,
                'data': {
                    'type': 'price_target',
                    'current_price': target_info['current_price'],
                    'target_price': target_info['target_price'],
                    'target_time': target_info['target_time'],
                    'direction': 'SIDEWAYS',
                    'movement_percent': 0,
                    'message': '当前市场无明显趋势'
                }
            })
        
        return jsonify({
            'success': True,
            'data': {
                'type': 'price_target',
                'current_price': target_info['current_price'],
                'target_price': target_info['target_price'],
                'target_time': target_info['target_time'],
                'direction': target_info['direction'],
                'movement_percent': target_info['movement_percent'],
                'is_calibrated': target_info.get('is_calibrated', False)
            }
        })
        
    except Exception as e:
        logger.error(f"价格目标预测失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def _predict_state(symbol: str, model_name: str):
    """市场状态预测（原有逻辑）"""
    try:
        # 加载模型
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': '模型文件不存在'}), 404
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # 查找 scaler
        base_name = model_name.replace('.pkl', '')
        scaler_path = os.path.join(MODEL_DIR, f"{base_name}_scaler.pkl")
        feature_path = os.path.join(MODEL_DIR, f"{base_name}_feature_cols.pkl")
        calibrator_path = os.path.join(MODEL_DIR, f"{base_name}_calibrators.pkl")
        
        # 尝试简化名称
        if not os.path.exists(scaler_path):
            parts = base_name.split('_')
            if len(parts) >= 2:
                simple_base = f"{parts[0]}_{parts[1]}"
                scaler_path = os.path.join(MODEL_DIR, f"{simple_base}_scaler.pkl")
                feature_path = os.path.join(MODEL_DIR, f"{simple_base}_feature_cols.pkl")
                calibrator_path = os.path.join(MODEL_DIR, f"{simple_base}_calibrators.pkl")
        
        if not os.path.exists(scaler_path):
            return jsonify({
                'success': False, 
                'error': '未找到对应的 scaler 文件，请重新训练状态模型',
                'debug': {'model': model_name, 'base': base_name, 'available': os.listdir(MODEL_DIR)}
            }), 400
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        feature_cols = None
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                feature_cols = pickle.load(f)
        
        calibrators = None
        if os.path.exists(calibrator_path):
            with open(calibrator_path, 'rb') as f:
                calibrators = pickle.load(f)
        
        # 获取数据（使用 H1）
        from core.utils.mt5 import get_rates_mt5
        import MetaTrader5 as mt5
        df = get_rates_mt5(symbol, mt5.TIMEFRAME_H1, 150)
        if df is None or df.empty:
            return jsonify({'success': False, 'error': '无法获取市场数据'}), 400
        
        # 计算特征
        df_feat = FeatureFactory.compute_features(df.tail(100))
        if len(df_feat) == 0:
            return jsonify({'success': False, 'error': '特征计算失败'}), 400
        
        latest = df_feat.iloc[-1:].copy()
        
        # 特征对齐
        if feature_cols:
            X = latest.reindex(columns=feature_cols, fill_value=0)
        elif hasattr(model, 'feature_names_in_'):
            X = latest.reindex(columns=list(model.feature_names_in_), fill_value=0)
        else:
            exclude = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume', 'real_volume', 'spread']
            cols = [c for c in latest.columns if c not in exclude and np.issubdtype(latest[c].dtype, np.number)]
            X = latest[cols]
        
        X_scaled = scaler.transform(X)
        
        # 预测
        raw_proba = model.predict_proba(X_scaled)[0]
        pred = int(model.predict(X_scaled)[0])
        
        if calibrators:
            calibrated_proba = []
            for i, calibrator in enumerate(calibrators):
                if calibrator is not None and i < len(raw_proba):
                    calibrated_proba.append(calibrator.predict([raw_proba[i]])[0])
                else:
                    calibrated_proba.append(raw_proba[i] if i < len(raw_proba) else 0.0)
            proba = np.array(calibrated_proba)
            proba = proba / (proba.sum() + 1e-10)
        else:
            proba = raw_proba
        
        confidence = float(max(proba))
        
        state_names = {0: '震荡', 1: '多头趋势', 2: '空头趋势', 3: '高波动突破'}
        state_emoji = {0: '🔄', 1: '📈', 2: '📉', 3: '⚡'}
        
        # 策略推荐
        recommendation = get_recommended_strategy(symbol, pred, "H1")
        
        return jsonify({
            'success': True,
            'data': {
                'type': 'state',
                'state': pred,
                'state_name': state_names.get(pred, '未知'),
                'state_emoji': state_emoji.get(pred, '❓'),
                'confidence': confidence,
                'probabilities': proba.tolist(),
                'current_price': float(df['close'].iloc[-1]),
                'timestamp': datetime.now().isoformat(),
                'recommended_strategy': recommendation.get('strategy', 'BOLLINGER_REVERSION'),
                'recommended_params': recommendation.get('params', {}),
                'recommendation_source': recommendation.get('source', 'unknown'),
                'recommendation_win_rate': recommendation.get('win_rate'),
                'recommendation_trades': recommendation.get('total_trades'),
                'recommendation_drawdown': recommendation.get('max_drawdown'),
                'recommendation_profit': recommendation.get('total_profit'),
                'is_calibrated': calibrators is not None
            }
        })
        
    except Exception as e:
        logger.error(f"状态预测失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============ 辅助 ============

@api_ai_predictor.route('/symbols', methods=['GET'])
def get_symbols():
    """获取可用商品列表"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'backtest_data')
    symbols = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('.csv'):
                parts = f.split('_')
                if len(parts) >= 1:
                    symbol = parts[0]
                    if symbol not in symbols:
                        symbols.append(symbol)
    return jsonify({'success': True, 'data': sorted(symbols)})


# ============ AI 机器人控制 ============

@api_ai_predictor.route('/bot/start', methods=['POST'])
def start_ai_bot():
    """启动 AI 机器人（仅支持状态模型）"""
    global _ai_bot_instance
    
    data = request.get_json()
    symbol = data.get('symbol', 'BTCUSDm')
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({'success': False, 'error': '请选择模型'}), 400
    
    # 检查是否为价格目标模型（机器人暂不支持）
    if 'price_target' in model_name.lower():
        return jsonify({'success': False, 'error': 'AI 机器人暂不支持价格目标模型，请使用状态模型'}), 400
    
    with _bot_lock:
        # 如果已有机器人运行，先停止
        if _ai_bot_instance and _ai_bot_instance.is_running:
            _ai_bot_instance.stop()
            time.sleep(1)
        
        try:
            from core.ai.ai_trading_bot import AITradingBot
            
            # 准备模型路径 - 支持灵活文件匹配
            model_path = os.path.join(MODEL_DIR, model_name)
            base_name = model_name.replace('.pkl', '')

            def find_file(base_name, suffix):
                """在 MODEL_DIR 中查找匹配的文件"""
                candidates = [
                    os.path.join(MODEL_DIR, f"{base_name}{suffix}"),
                    os.path.join(MODEL_DIR, f"{base_name.replace('_200', '')}{suffix}"),
                    os.path.join(MODEL_DIR, f"{base_name.replace('_100', '')}{suffix}"),
                    os.path.join(MODEL_DIR, f"{base_name.replace('_60', '')}{suffix}"),
                    os.path.join(MODEL_DIR, f"{base_name.replace('_30', '')}{suffix}"),
                ]
                # 添加只用 symbol+timeframe 的变体
                parts = base_name.split('_')
                if len(parts) >= 2:
                    simple_base = f"{parts[0]}_{parts[1]}"
                    candidates.append(os.path.join(MODEL_DIR, f"{simple_base}{suffix}"))
                for candidate in candidates:
                    if os.path.exists(candidate):
                        return candidate
                return None

            scaler_path = find_file(base_name, "_scaler.pkl")
            feature_path = find_file(base_name, "_feature_cols.pkl")
            calibrator_path = find_file(base_name, "_calibrators.pkl")

            if not os.path.exists(model_path):
                return jsonify({'success': False, 'error': f'模型文件不存在: {model_path}'}), 400
            if not scaler_path:
                return jsonify({'success': False, 'error': f'Scaler文件不存在，请重新训练状态模型'}), 400
            
            # 创建机器人
            _ai_bot_instance = AITradingBot(
                symbol=symbol,
                model_path=model_path,
                scaler_path=scaler_path,
                feature_cols_path=feature_path if feature_path else None,
                calibrator_path=calibrator_path if calibrator_path else None
            )
            
            success, msg = _ai_bot_instance.start()
            if success:
                return jsonify({'success': True, 'message': msg, 'bot_id': id(_ai_bot_instance)})
            else:
                return jsonify({'success': False, 'error': msg}), 500
                
        except ImportError as e:
            return jsonify({'success': False, 'error': f'导入AI机器人模块失败: {e}'}), 500
        except Exception as e:
            logger.error(f"启动AI机器人失败: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500


@api_ai_predictor.route('/bot/stop', methods=['POST'])
def stop_ai_bot():
    """停止 AI 机器人"""
    global _ai_bot_instance
    
    with _bot_lock:
        if _ai_bot_instance is None:
            return jsonify({'success': False, 'error': '机器人未启动'}), 400
        
        success, msg = _ai_bot_instance.stop()
        if success:
            return jsonify({'success': True, 'message': msg})
        else:
            return jsonify({'success': False, 'error': msg}), 500


@api_ai_predictor.route('/bot/status', methods=['GET'])
def get_bot_status():
    """获取 AI 机器人状态"""
    global _ai_bot_instance
    
    with _bot_lock:
        if _ai_bot_instance is None:
            return jsonify({
                'success': True,
                'data': {'is_running': False, 'message': '未启动'}
            })
        return jsonify({
            'success': True,
            'data': _ai_bot_instance.get_status()
        })


@api_ai_predictor.route('/bot/logs/stream')
def stream_bot_logs():
    """SSE 推送机器人日志"""
    global _ai_bot_instance
    
    def generate():
        last_count = 0
        while True:
            if _ai_bot_instance:
                logs = _ai_bot_instance.logs
                if len(logs) > last_count:
                    for log in logs[last_count:]:
                        yield f"data: {json.dumps(log)}\n\n"
                    last_count = len(logs)
            time.sleep(1)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

# ============ AI 回测 ============

_backtest_status = {
    'running': False,
    'progress': 0,
    'message': '',
    'done': False,
    'result': None,
    'logs': []
}

@api_ai_predictor.route('/backtest/run', methods=['POST'])
def run_backtest():
    """启动 AI 回测（异步）"""
    global _backtest_status
    
    data = request.get_json()
    symbol = data.get('symbol', 'XAUUSDm')
    model_name = data.get('model_name')
    timeframe = data.get('timeframe', 'M5')
    data_file = data.get('data_file')
    
    if not model_name:
        return jsonify({'success': False, 'error': '请选择模型'}), 400
    
    # 检查是否为价格目标模型
    is_price_target = 'price_target' in model_name.lower()
    
    # 重置状态
    _backtest_status = {
        'running': True,
        'progress': 0,
        'message': '准备回测数据...',
        'done': False,
        'result': None,
        'logs': [],
        'is_price_target': is_price_target
    }
    
    def run_backtest_thread():
        global _backtest_status
        try:
            from core.ai.ai_backtester import AIBacktester
            import pandas as pd
            
            # 1. 查找数据文件
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'backtest_data')
            
            if data_file:
                file_path = data_file
            else:
                file_path = os.path.join(data_dir, f'{symbol}_{timeframe}_data.csv')
            
            if not os.path.exists(file_path):
                _backtest_status['running'] = False
                _backtest_status['done'] = True
                _backtest_status['error'] = f'数据文件不存在: {file_path}'
                _backtest_status['message'] = '数据文件不存在'
                return
            
            _add_backtest_log(f'📂 加载数据: {file_path}')
            _backtest_status['progress'] = 10
            
            df = pd.read_csv(file_path, parse_dates=['time'])
            _add_backtest_log(f'📊 加载了 {len(df)} 根 K 线')
            _backtest_status['progress'] = 20
            
            # 2. 查找模型文件
            model_path = os.path.join(MODEL_DIR, model_name)
            base_name = model_name.replace('.pkl', '')
            
            def find_file(suffix):
                candidates = [
                    os.path.join(MODEL_DIR, f"{base_name}{suffix}"),
                    os.path.join(MODEL_DIR, f"{base_name.replace('_200', '')}{suffix}"),
                    os.path.join(MODEL_DIR, f"{base_name.replace('_100', '')}{suffix}"),
                    os.path.join(MODEL_DIR, f"{base_name.replace('_60', '')}{suffix}"),
                ]
                parts = base_name.split('_')
                if len(parts) >= 2:
                    simple_base = f"{parts[0]}_{parts[1]}"
                    candidates.append(os.path.join(MODEL_DIR, f"{simple_base}{suffix}"))
                for c in candidates:
                    if os.path.exists(c):
                        return c
                return None
            
            scaler_path = find_file("_scaler.pkl") if not is_price_target else None
            feature_path = find_file("_feature_cols.pkl") if not is_price_target else None
            calibrator_path = find_file("_calibrators.pkl") if not is_price_target else None
            price_target_path = find_file("_price_target.pkl") if is_price_target else None
            
            _add_backtest_log(f'🧠 加载模型: {model_name}')
            _backtest_status['progress'] = 30
            
            # 3. 创建回测器
            backtester = AIBacktester(
                symbol=symbol,
                model_path=model_path,
                scaler_path=scaler_path,
                feature_cols_path=feature_path,
                calibrator_path=calibrator_path,
                price_target_model_path=price_target_path
            )
            
            # 根据模型类型设置目标预测
            backtester.use_target_prediction = is_price_target
            
            _add_backtest_log('🚀 开始回测...')
            _backtest_status['progress'] = 40
            
            # 4. 运行回测
            report = backtester.run(df)
            
            _backtest_status['progress'] = 90
            _add_backtest_log('✅ 回测完成，生成报告...')
            
            # 5. 处理结果
            _backtest_status['result'] = {
                'symbol': report['symbol'],
                'initial_capital': report['initial_capital'],
                'final_capital': report['final_capital'],
                'total_profit': report['total_profit'],
                'total_profit_percent': report['total_profit_percent'],
                'total_trades': report['total_trades'],
                'winning_trades': report['winning_trades'],
                'losing_trades': report['losing_trades'],
                'win_rate': report['win_rate'],
                'max_drawdown': report['max_drawdown'],
                'switches_count': report['switches_count'],
                'switches_by_strategy': report['switches_by_strategy'],
                'target_stats': report.get('target_stats', {'total': 0, 'by_direction': {}}),
                'avg_target_time': report.get('avg_target_time', 0),
                'equity_curve': report['equity_curve'],
                'trades': report['trade_log'],
                'strategy_switches': report['strategy_switches'][-20:],  # 最近20条
                'use_target_prediction': report.get('use_target_prediction', False)
            }
            
            _backtest_status['running'] = False
            _backtest_status['done'] = True
            _backtest_status['progress'] = 100
            _backtest_status['message'] = '回测完成'
            _add_backtest_log('📊 回测完成！查看结果')
            
        except Exception as e:
            logger.error(f"回测失败: {e}")
            _backtest_status['running'] = False
            _backtest_status['done'] = True
            _backtest_status['error'] = str(e)
            _backtest_status['message'] = f'回测失败: {e}'
            _add_backtest_log(f'❌ 回测失败: {e}')
    
    def _add_backtest_log(msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        _backtest_status['logs'].append({
            'timestamp': timestamp,
            'message': msg
        })
    
    thread = threading.Thread(target=run_backtest_thread, daemon=True)
    thread.start()
    
    return jsonify({'success': True, 'message': '回测已启动'})


@api_ai_predictor.route('/backtest/status', methods=['GET'])
def get_backtest_status():
    """获取回测状态"""
    return jsonify({
        'success': True,
        'data': _backtest_status
    })


@api_ai_predictor.route('/backtest/stream')
def stream_backtest_logs():
    """SSE 推送回测日志"""
    def generate():
        last_count = 0
        while True:
            logs = _backtest_status.get('logs', [])
            if len(logs) > last_count:
                for log in logs[last_count:]:
                    yield f"data: {json.dumps(log)}\n\n"
                last_count = len(logs)
            
            if _backtest_status.get('done', False):
                break
            
            time.sleep(0.5)
        
        # 发送完成信号
        yield f"data: {json.dumps({'done': True, 'message': '回测完成'})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@api_ai_predictor.route('/backtest/cancel', methods=['POST'])
def cancel_backtest():
    """取消回测"""
    global _backtest_status
    if _backtest_status.get('running', False):
        _backtest_status['running'] = False
        _backtest_status['done'] = True
        _backtest_status['message'] = '回测已取消'
        _backtest_status['logs'].append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'message': '🛑 回测已取消'
        })
        return jsonify({'success': True, 'message': '已取消回测'})
    return jsonify({'success': False, 'error': '没有正在运行的回测'})