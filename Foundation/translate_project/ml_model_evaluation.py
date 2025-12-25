"""
机器学习模型评估模块
Model Evaluation for Machine Learning

Java对应实现：可以使用Weka的Evaluation类
Java equivalent: Use Weka's Evaluation class
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, KFold

# 配置中文字体支持
from ml_font_config import setup_chinese_font
setup_chinese_font()

# Java对应：import weka.classifiers.Evaluation;


class ModelEvaluator:
    """
    模型评估类
    
    Java对应实现：
    public class ModelEvaluator {
        private double[] yTrue;
        private double[] yPred;
        
        public ModelEvaluator(double[] yTrue, double[] yPred) {
            this.yTrue = yTrue;
            this.yPred = yPred;
        }
        
        // 计算准确率
        public double calculateAccuracy() {
            int correct = 0;
            for (int i = 0; i < yTrue.length; i++) {
                if (yTrue[i] == yPred[i]) {
                    correct++;
                }
            }
            return (double) correct / yTrue.length;
        }
    }
    """
    
    def __init__(self):
        """
        初始化评估器
        
        Java对应：
        public ModelEvaluator() {
            // 初始化
        }
        """
        pass
    
    def classification_metrics(self, y_true, y_pred, average='binary'):
        """
        计算分类指标
        
        Java对应：
        public ClassificationMetrics calculateMetrics(int[] yTrue, int[] yPred) {
            // 计算混淆矩阵
            int[][] confusionMatrix = calculateConfusionMatrix(yTrue, yPred);
            
            int tp = confusionMatrix[1][1];  // True Positive
            int tn = confusionMatrix[0][0];  // True Negative
            int fp = confusionMatrix[0][1];  // False Positive
            int fn = confusionMatrix[1][0];  // False Negative
            
            // 准确率 Accuracy = (TP + TN) / (TP + TN + FP + FN)
            double accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
            
            // 精确率 Precision = TP / (TP + FP)
            double precision = (double)tp / (tp + fp);
            
            // 召回率 Recall = TP / (TP + FN)
            double recall = (double)tp / (tp + fn);
            
            // F1分数 F1 = 2 * (Precision * Recall) / (Precision + Recall)
            double f1 = 2 * (precision * recall) / (precision + recall);
            
            return new ClassificationMetrics(accuracy, precision, recall, f1);
        }
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            average: 多分类时的平均方式
        Returns:
            各项指标的字典
        """
        # 准确率：正确预测的比例
        # Java: accuracy = (double)correctPredictions / totalPredictions
        accuracy = accuracy_score(y_true, y_pred)
        
        # 精确率：预测为正的样本中真正为正的比例
        # Java: precision = (double)truePositive / (truePositive + falsePositive)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        
        # 召回率：真正为正的样本中被预测为正的比例
        # Java: recall = (double)truePositive / (truePositive + falseNegative)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        
        # F1分数：精确率和召回率的调和平均
        # Java: f1 = 2 * (precision * recall) / (precision + recall)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print("分类指标:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, show=True):
        """
        绘制混淆矩阵
        
        Java对应：
        public void plotConfusionMatrix(int[] yTrue, int[] yPred) {
            int[][] cm = calculateConfusionMatrix(yTrue, yPred);
            
            // 使用JFreeChart绘制热力图
            DefaultCategoryDataset dataset = new DefaultCategoryDataset();
            for (int i = 0; i < cm.length; i++) {
                for (int j = 0; j < cm[i].length; j++) {
                    dataset.addValue(cm[i][j], "True " + i, "Pred " + j);
                }
            }
            
            JFreeChart chart = ChartFactory.createHeatMap(
                "Confusion Matrix", "Predicted", "Actual", dataset);
        }
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 类别标签
            show: 是否显示图形（默认True）
        """
        # 计算混淆矩阵
        # Java: cm = calculateConfusionMatrix(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"混淆矩阵:\n{cm}")
        
        # 绘制热力图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        
        if show:
            print("正在显示混淆矩阵图形...")
            plt.show()
            print("混淆矩阵图形已显示")
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, show=True):
        """
        绘制ROC曲线
        
        Java对应：
        public void plotROCCurve(int[] yTrue, double[] yPredProba) {
            // 计算不同阈值下的TPR和FPR
            List<Point2D> rocPoints = new ArrayList<>();
            
            for (double threshold = 0.0; threshold <= 1.0; threshold += 0.01) {
                int tp = 0, fp = 0, tn = 0, fn = 0;
                
                for (int i = 0; i < yTrue.length; i++) {
                    int pred = yPredProba[i] >= threshold ? 1 : 0;
                    if (yTrue[i] == 1 && pred == 1) tp++;
                    else if (yTrue[i] == 0 && pred == 1) fp++;
                    else if (yTrue[i] == 0 && pred == 0) tn++;
                    else fn++;
                }
                
                double tpr = (double)tp / (tp + fn);  // True Positive Rate
                double fpr = (double)fp / (fp + tn);  // False Positive Rate
                
                rocPoints.add(new Point2D.Double(fpr, tpr));
            }
            
            // 计算AUC
            double auc = calculateAUC(rocPoints);
            
            // 使用JFreeChart绘制ROC曲线
            XYSeries series = new XYSeries("ROC Curve (AUC = " + auc + ")");
            for (Point2D point : rocPoints) {
                series.add(point.getX(), point.getY());
            }
        }
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            show: 是否显示图形（默认True）
        """
        # 计算ROC曲线的点
        # Java: 需要手动计算不同阈值下的TPR和FPR
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        # 计算AUC（曲线下面积）
        # Java: auc = calculateAUC(fpr, tpr)
        roc_auc = auc(fpr, tpr)
        
        print(f"AUC分数: {roc_auc:.4f}")
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='随机猜测')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if show:
            print("正在显示ROC曲线图形...")
            plt.show()
            print("ROC曲线图形已显示")
        
        return roc_auc
    
    def regression_metrics(self, y_true, y_pred):
        """
        计算回归指标
        
        Java对应：
        public RegressionMetrics calculateRegressionMetrics(double[] yTrue, double[] yPred) {
            int n = yTrue.length;
            
            // MSE = (1/n) * Σ(y_true - y_pred)²
            double mse = 0.0;
            for (int i = 0; i < n; i++) {
                double error = yTrue[i] - yPred[i];
                mse += error * error;
            }
            mse /= n;
            
            // RMSE = √MSE
            double rmse = Math.sqrt(mse);
            
            // MAE = (1/n) * Σ|y_true - y_pred|
            double mae = 0.0;
            for (int i = 0; i < n; i++) {
                mae += Math.abs(yTrue[i] - yPred[i]);
            }
            mae /= n;
            
            // R² = 1 - (SS_res / SS_tot)
            double yMean = Arrays.stream(yTrue).average().getAsDouble();
            double ssTot = 0.0, ssRes = 0.0;
            
            for (int i = 0; i < n; i++) {
                ssTot += Math.pow(yTrue[i] - yMean, 2);
                ssRes += Math.pow(yTrue[i] - yPred[i], 2);
            }
            
            double r2 = 1 - (ssRes / ssTot);
            
            return new RegressionMetrics(mse, rmse, mae, r2);
        }
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        Returns:
            各项指标的字典
        """
        # 均方误差 MSE
        # Java: mse = sum((y_true - y_pred)²) / n
        mse = mean_squared_error(y_true, y_pred)
        
        # 均方根误差 RMSE
        # Java: rmse = Math.sqrt(mse)
        rmse = np.sqrt(mse)
        
        # 平均绝对误差 MAE
        # Java: mae = sum(|y_true - y_pred|) / n
        mae = mean_absolute_error(y_true, y_pred)
        
        # R²决定系数
        # Java: r2 = 1 - (SS_res / SS_tot)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }
        
        print("回归指标:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def cross_validation(self, model, X, y, cv=5):
        """
        交叉验证
        
        Java对应：
        public double[] crossValidation(Classifier model, Instances data, int folds) 
                throws Exception {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(model, data, folds, new Random(1));
            
            double[] scores = new double[folds];
            // 获取每折的准确率
            for (int i = 0; i < folds; i++) {
                scores[i] = eval.pctCorrect() / 100.0;
            }
            
            return scores;
        }
        
        Args:
            model: 模型对象
            X: 特征矩阵
            y: 标签
            cv: 折数
        Returns:
            交叉验证分数
        """
        # K折交叉验证
        # Java: 使用Weka的crossValidateModel方法
        scores = cross_val_score(model, X, y, cv=cv)
        
        print(f"\n{cv}折交叉验证结果:")
        print("-" * 40)
        print(f"各折分数: {scores}")
        print(f"平均分数: {scores.mean():.4f}")
        print(f"标准差: {scores.std():.4f}")
        
        return scores
    
    def plot_learning_curve(self, train_sizes, train_scores, val_scores):
        """
        绘制学习曲线
        
        Java对应：
        public void plotLearningCurve(int[] trainSizes, double[] trainScores, 
                                     double[] valScores) {
            XYSeries trainSeries = new XYSeries("Training Score");
            XYSeries valSeries = new XYSeries("Validation Score");
            
            for (int i = 0; i < trainSizes.length; i++) {
                trainSeries.add(trainSizes[i], trainScores[i]);
                valSeries.add(trainSizes[i], valScores[i]);
            }
            
            XYSeriesCollection dataset = new XYSeriesCollection();
            dataset.addSeries(trainSeries);
            dataset.addSeries(valSeries);
            
            JFreeChart chart = ChartFactory.createXYLineChart(
                "Learning Curve", "Training Size", "Score", dataset);
        }
        
        Args:
            train_sizes: 训练集大小
            train_scores: 训练分数
            val_scores: 验证分数
        """
        plt.figure(figsize=(10, 6))
        
        # 计算均值和标准差
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # 绘制训练分数
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='训练分数')
        plt.fill_between(train_sizes, train_mean - train_std, 
                        train_mean + train_std, alpha=0.1, color='r')
        
        # 绘制验证分数
        plt.plot(train_sizes, val_mean, 'o-', color='g', label='验证分数')
        plt.fill_between(train_sizes, val_mean - val_std, 
                        val_mean + val_std, alpha=0.1, color='g')
        
        plt.xlabel('训练样本数')
        plt.ylabel('分数')
        plt.title('学习曲线')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()


def example_usage():
    """
    使用示例
    
    Java对应：
    public static void main(String[] args) throws Exception {
        // 生成示例数据
        Random random = new Random(42);
        double[][] X = new double[100][2];
        int[] y = new int[100];
        
        for (int i = 0; i < 100; i++) {
            X[i][0] = random.nextGaussian();
            X[i][1] = random.nextGaussian();
            y[i] = (X[i][0] + X[i][1] > 0) ? 1 : 0;
        }
        
        // 训练模型
        Logistic model = new Logistic();
        Instances data = createInstances(X, y);
        model.buildClassifier(data);
        
        // 预测
        int[] predictions = new int[100];
        for (int i = 0; i < 100; i++) {
            predictions[i] = (int)model.classifyInstance(data.instance(i));
        }
        
        // 评估
        ModelEvaluator evaluator = new ModelEvaluator();
        evaluator.classificationMetrics(y, predictions);
    }
    """
    print("=" * 50)
    print("模型评估示例")
    print("=" * 50)
    
    # 生成示例数据
    np.random.seed(42)
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    # 生成分类数据
    # Java: 手动生成或使用数据生成工具
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=15, n_redundant=5,
                               random_state=42)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    # Java: Logistic model = new Logistic()
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 创建评估器
    # Java: ModelEvaluator evaluator = new ModelEvaluator()
    evaluator = ModelEvaluator()
    
    print("\n1. 分类指标")
    print("-" * 50)
    # Java: evaluator.classificationMetrics(y_test, y_pred)
    metrics = evaluator.classification_metrics(y_test, y_pred)
    
    print("\n2. 混淆矩阵")
    print("-" * 50)
    # Java: evaluator.plotConfusionMatrix(y_test, y_pred)
    cm = evaluator.plot_confusion_matrix(y_test, y_pred, labels=['类别0', '类别1'])
    
    print("\n3. ROC曲线")
    print("-" * 50)
    # Java: evaluator.plotROCCurve(y_test, y_pred_proba)
    roc_auc = evaluator.plot_roc_curve(y_test, y_pred_proba)
    
    print("\n4. 交叉验证")
    print("-" * 50)
    # Java: evaluator.crossValidation(model, data, 5)
    cv_scores = evaluator.cross_validation(model, X_train, y_train, cv=5)
    
    print("\n5. 分类报告")
    print("-" * 50)
    # Java: 需要手动计算各类别的precision, recall, f1-score
    print(classification_report(y_test, y_pred, 
                               target_names=['类别0', '类别1']))
    
    # 回归示例
    print("\n" + "=" * 50)
    print("回归模型评估示例")
    print("=" * 50)
    
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    
    # 生成回归数据
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, 
                                   noise=10, random_state=42)
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)
    
    # 训练回归模型
    reg_model = LinearRegression()
    reg_model.fit(X_train_reg, y_train_reg)
    
    # 预测
    y_pred_reg = reg_model.predict(X_test_reg)
    
    print("\n回归指标:")
    print("-" * 50)
    # Java: evaluator.regressionMetrics(y_test_reg, y_pred_reg)
    reg_metrics = evaluator.regression_metrics(y_test_reg, y_pred_reg)
    
    # 绘制预测vs真实值
    print("\n绘制回归预测结果图...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
    plt.plot([y_test_reg.min(), y_test_reg.max()],
            [y_test_reg.min(), y_test_reg.max()],
            'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('回归预测结果')
    plt.grid(True)
    print("正在显示回归预测结果图形...")
    plt.show()
    print("回归预测结果图形已显示")


if __name__ == "__main__":
    """
    Java对应：
    public static void main(String[] args) {
        exampleUsage();
    }
    """
    example_usage()