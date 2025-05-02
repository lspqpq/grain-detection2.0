import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
from skimage.feature import local_binary_pattern
from datetime import datetime

# 中文标注支持函数
def put_chinese_text(image, text, position, color, font_size=22):
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except:
        font = ImageFont.load_default()
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    # 用 textbbox 替代 textsize
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = position[0] - text_width // 2
    y = position[1] - text_height // 2
    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# 预处理流程
def enhanced_preprocessing(img, params):

    # 中值滤波降噪
    median = cv2.medianBlur(img, params['median_kernel'])
    
    # 边缘检测（前置）
    edges = cv2.Canny(median, params['canny_low'], params['canny_high'])
    
    # 多通道分割
    lab = cv2.cvtColor(median, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    # 基于亮度的自适应阈值
    _, l_thresh = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # 形态学优化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    refined_mask = cv2.morphologyEx(l_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 背景移除示例（假设背景为白色）
    bg_mask = cv2.bitwise_not(refined_mask)
    bg_removed = cv2.bitwise_and(median, median, mask=bg_mask)

    return {
        'median': median,
        'edges': edges,
        'refined_mask': refined_mask,
        'bg_mask': bg_mask,          # 新增
        'bg_removed': bg_removed     # 新增
    }

# 多通道分割颗粒
def multi_channel_segmentation(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 各通道处理
    l_channel = lab[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    
    # 自适应阈值处理
    _, l_thresh = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, s_thresh = cv2.threshold(s_channel, 30, 255, cv2.THRESH_BINARY)
    _, v_thresh = cv2.threshold(v_channel, 150, 255, cv2.THRESH_BINARY)
    
    # 组合掩模
    combined_mask = cv2.bitwise_and(l_thresh, cv2.bitwise_and(s_thresh, v_thresh))
    
    # 形态学优化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    refined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return {
        'l_channel': l_channel,
        'l_thresh': l_thresh,
        's_thresh': s_thresh,
        'v_thresh': v_thresh,
        'combined_mask': combined_mask,
        'refined_mask': refined_mask
    }

# 增强特征提取函数
def extract_features(contour, img, edges):
    features = {}
    img_h, img_w = img.shape[:2]
    
    # 椭圆拟合分析
    try:
        ellipse = cv2.fitEllipse(contour)
        (x,y), (major_axis, minor_axis), angle = ellipse
        features['ellipse_ratio'] = major_axis / minor_axis if minor_axis != 0 else 1.0
        features['ellipse_area'] = np.pi * (major_axis/2) * (minor_axis/2)
    except:
        features['ellipse_ratio'] = 1.0
        features['ellipse_area'] = 0
    
    # 凸包缺陷分析
    hull = cv2.convexHull(contour, returnPoints=False)
    defects_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    features['convex_defects'] = 0
    if hull is not None and len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is not None:
            features['convex_defects'] = len(defects)
    
    # 颜色特征（LAB空间内部差异）
    mask = np.zeros((img_h, img_w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    features['a_std'] = np.std(lab[:,:,1][mask==255])
    features['b_std'] = np.std(lab[:,:,2][mask==255])
    
    # 边缘凹陷分析
    edge_mask = cv2.bitwise_and(edges, edges, mask=mask)
    features['edge_density'] = np.sum(edge_mask) / cv2.contourArea(contour)
    
    # 纹理分析
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    features['texture_var'] = np.var(lbp[mask==255])
    
    return features

# 分类逻辑
def classify_defect(features, params):
    # 未熟粒：椭圆比率大（扁平）
    if features['ellipse_ratio'] > params['immature_ratio']:
        return "未熟粒"
    
    # 虫蚀粒：边缘密度高且有凸缺陷
    if (features['edge_density'] > params['wormhole_edge'] and 
        features['convex_defects'] >= params['wormhole_defects']):
        return "虫蚀粒"
    
    # 病斑粒：颜色差异大
    if (features['a_std'] > params['disease_a'] or 
        features['b_std'] > params['disease_b']):
        return "病斑粒"
    
    # 生芽粒：椭圆拟合良好且边缘外凸
    if (params['sprout_ratio_min'] < features['ellipse_ratio'] < params['sprout_ratio_max'] and
        features['edge_density'] < params['sprout_edge']):
        return "生芽粒"
    
    # 霉变粒：纹理差异大
    if features['texture_var'] > params['mold_texture']:
        return "霉变粒"
    
    return None

# Streamlit界面配置
st.set_page_config(layout="wide")
st.title("🌾WTG糙米不完善粒检测系统")

with st.sidebar:
    st.header("参数设置")
    
    # 预处理参数
    st.subheader("预处理设置")
    params = {
        'median_kernel': st.selectbox("中值滤波核大小", [3, 5, 7], index=1),
        'canny_low': st.slider("Canny低阈值", 30, 100, 50),
        'canny_high': st.slider("Canny高阈值", 100, 200, 150),
        'min_area': st.number_input("最小颗粒面积", 50, 10000, 200),
        'max_area': st.number_input("最大颗粒面积", 1000, 50000, 10000)
    }

    # 分类参数
    st.subheader("未熟粒参数")
    params.update({
        'immature_ratio': st.slider("椭圆比率阈值", 1.5, 3.0, 2.2)
    })
    st.subheader("虫蚀粒参数")
    params.update({
        'wormhole_edge': st.slider("边缘密度阈值", 0.1, 0.5, 0.3),
        'wormhole_defects': st.slider("凸缺陷阈值", 1, 5, 2)
    })
    st.subheader("病斑粒参数")
    params.update({
        'disease_a': st.slider("A通道标准差", 10, 30, 15),
        'disease_b': st.slider("B通道标准差", 10, 30, 18)
    })
    st.subheader("生芽粒参数")
    params.update({
        'sprout_ratio_min': st.slider("椭圆比率下限", 1.0, 2.0, 1.5),
        'sprout_ratio_max': st.slider("椭圆比率上限", 2.0, 3.0, 2.5),
        'sprout_edge': st.slider("最大边缘密度", 0.1, 0.4, 0.2)
    })
    st.subheader("霉变粒参数")
    params.update({
        'mold_texture': st.slider("纹理方差阈值", 1000, 3000, 1500)
    })

def process_image(img, params):
    # 预处理
    preprocessed = enhanced_preprocessing(img, params)
    
    # 分割流程
    segmentation = multi_channel_segmentation(img)
    process_images = {
        'segmentation': segmentation
    }
    
    # 轮廓检测
    contours, _ = cv2.findContours(preprocessed['refined_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 特征分析与分类
    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (params['min_area'] < area < params['max_area']):
            continue
        
        features = extract_features(cnt, img, preprocessed['edges'])
        defect_type = classify_defect(features, params)
        
        if defect_type:
            defects.append({
                'contour': cnt,
                'type': defect_type,
                'features': features
            })
    
    return {
        'defects': defects,
        'preprocessed': preprocessed,
        'process_images': process_images
    }

# 主程序
input_method = st.radio("选择输入方式：", 
                       ["上传图片文件", "使用相机拍摄"],
                       horizontal=True,
                       label_visibility="visible")

uploaded_file = None
if input_method == "上传图片文件":
    uploaded_file = st.file_uploader(
        "上传糙米图像", 
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed",
        key="main_upload"
    )
else:
    uploaded_file = st.camera_input(
        "即时拍摄检测", 
        label_visibility="collapsed",
        help="请将摄像头对准需要检测的糙米样本",
        key="main_camera"
    )

if uploaded_file:
    # 统一处理上传文件和相机拍摄的图片
    image = Image.open(uploaded_file)
    
    # 处理手机拍摄图片的方向问题
    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass
    
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = process_image(img, params)
    
    # ================== 实时显示检测结果 ==================
    st.header("检测结果")
    
    # 创建可视化图像
    result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color_map = {
        "未熟粒": (255, 150, 0),    # 橙色
        "虫蚀粒": (0, 255, 255),    # 黄色
        "病斑粒": (0, 0, 255),      # 红色
        "生芽粒": (255, 0, 255),    # 紫色
        "霉变粒": (0, 255, 0)       # 绿色
    }
    
    # 绘制检测结果
    for defect in results['defects']:
        contour = defect['contour']
        defect_type = defect['type']
        color = color_map[defect_type]
        
        # 绘制椭圆拟合
        try:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(result_img, ellipse, color, 4)
        except:
            cv2.drawContours(result_img, [contour], -1, color, 4)
        
        # 添加中文标注
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            result_img = put_chinese_text(
                cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR),
                defect_type,
                (cX, cY+45),
                color,
                font_size=25
            )
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    # 显示结果
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.image(result_img, use_container_width=True)
    
    with col2:
        # 右边栏统计
        defect_counts = {k:0 for k in color_map.keys()}
        for d in results['defects']:
            defect_counts[d['type']] += 1
        
        st.subheader("不完善粒检测统计")
        for t in color_map.keys():
            st.markdown(f"""
        <div style="padding:12px; margin:8px 0; 
                   background:#f8f9fa; 
                   border-radius:8px;
                   box-shadow:0 2px 4px rgba(0,0,0,0.05);">
            <div style="font-size:18px; color:#666;">{t}</div>
            <div style="font-size:28px; font-weight:bold; color:#333;">{defect_counts[t]}</div>
        </div>
        """, unsafe_allow_html=True)
        
         # 统计结果容器
        stats_container = st.container()
        with stats_container:
            # 添加自定义CSS样式
            st.markdown("""
            <style>
            .stats-card {
                padding: 12px;
                margin: 8px 0;
                border-radius: 8px;
                background: #f8f9fa;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stats-title {
                font-size: 1.1em;
                color: #2c3e50;
                margin-bottom: 10px;
            }
            </style>
            """, unsafe_allow_html=True)
            
        # 质量评级
            total_defects = sum(defect_counts.values())
            if total_defects < 5:
                quality_level = "优"
                bg_color = "#e6f4ea"
                font_color = "#2e8540"
            elif total_defects < 10:
                quality_level = "良"
                bg_color = "#fff3cd"
                font_color = "#856404"
            else:
                quality_level = "差"
                bg_color = "#f8d7da"
                font_color = "#721c24"
            
            st.markdown(f"""
    <div class="stats-card" style="background:{bg_color};">
        <div class="stats-title" style="font-size:26px;">质量评级</div>
        <div style="text-align:center; padding:18px 0;">
            <div style="font-size:46px; font-weight:bold; color:{font_color}; 
                     line-height:1.2; margin:10px 0;">{quality_level}</div>
            <div style="font-size:20px; color:{font_color};">
                不完善粒总数：{total_defects}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

     # ================== 处理流程分析 ==================
    with st.expander("查看详细图像处理流程"):
        st.subheader("预处理效果")
        cols = st.columns(3)
        cols[0].image(image, caption="原始图像")
        cols[1].image(cv2.cvtColor(results['preprocessed']['median'], cv2.COLOR_BGR2RGB),
                      caption="中值滤波结果")
        cols[2].image(results['preprocessed']['edges'], 
                      caption="边缘检测结果", clamp=True)

    # ================== 分割流程可视化 ==================
        st.subheader("分割流程")
        seg = results['process_images']['segmentation']
    
        col1, col2, col3 = st.columns(3)
        col1.image(seg['l_thresh'], caption="亮度通道阈值", clamp=True)
        col2.image(seg['s_thresh'], caption="饱和度通道阈值", clamp=True)
        col3.image(seg['v_thresh'], caption="明度通道阈值", clamp=True)
    
        col4, col5 = st.columns(2)
        col4.image(seg['combined_mask'], caption="组合掩模", clamp=True)
        col5.image(seg['refined_mask'], caption="优化后掩模", clamp=True)

        # 只有有缺陷时才分析特征
        if results['defects']:
            sample = results['defects'][0]
            st.subheader("不完善颗粒特征分析")
            
            feature_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.drawContours(feature_img, [sample['contour']], -1, (0,255,0), 2)
            
            cols = st.columns(3)
            cols[0].image(feature_img, caption="轮廓分析")
            
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.drawContours(mask, [sample['contour']], -1, 255, -1)
            color_analysis = cv2.bitwise_and(img, img, mask=mask)
            cols[1].image(cv2.cvtColor(color_analysis, cv2.COLOR_BGR2RGB), 
                          caption="颜色分析")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            texture_vis = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
            cols[2].image(texture_vis, caption="纹理分析", clamp=True)

    
    # ================== 导出功能 ==================
    st.markdown("---")
    st.subheader("结果导出")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    col_export1, col_export2 = st.columns(2)
    with col_export1:
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        img_bytes = cv2.imencode('.png', result_img_rgb)[1].tobytes()
        st.download_button(
            label="下载检测结果图片",
            data=img_bytes,
            file_name=f"检测结果_{timestamp}.png",
            mime="image/png"
        )
    with col_export2:
        text_content = f"糙米不完善粒检测报告\n生成时间：{timestamp}\n\n"
        text_content += "="*30 + "\n"
        text_content += "缺陷类型\t数量\n"
        text_content += "-"*30 + "\n"
        for t, c in defect_counts.items():
            text_content += f"{t}\t{c}\n"
        text_content += "="*30 + "\n"
        text_content += f"总计\t{total_defects}"
        st.download_button(
            label="下载检测报告",
            data=text_content.encode('utf-8'),
            file_name=f"检测报告_{timestamp}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    st.markdown("---")
    st.caption("Developed by LSQ | 检测版本4.0")
