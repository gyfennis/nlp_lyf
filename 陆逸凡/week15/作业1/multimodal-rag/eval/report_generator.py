"""HTML report generator for RAG evaluation results."""
from typing import Dict, List


def generate_table_rows(results: List[Dict]) -> str:
    """Generate HTML table rows from evaluation results."""
    rows = []
    for r in results:
        score_class = "score-good" if r["total_score"] >= 0.75 else "score-bad"
        rows.append(f"""
            <tr>
                <td>{r['query_id']}</td>
                <td>{r['filename_score']:.2f}</td>
                <td>{r['page_score']:.2f}</td>
                <td>{r['content_score']:.2f}</td>
                <td class="{score_class}">{r['total_score']:.2f}</td>
                <td><details><summary>查看答案</summary>{escape_html(r.get('answer', ''))[:200]}...</details></td>
            </tr>
        """)
    return "".join(rows)


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def generate_failure_analysis(results: List[Dict]) -> str:
    """Generate failure analysis section."""
    failures = [r for r in results if r["total_score"] < 0.5]

    if not failures:
        return "<p>所有测试用例均达到最低分数要求。</p>"

    rows = []
    for r in failures:
        rows.append(f"""
            <li>
                <strong>{r['query_id']}</strong>:
                文件名{r['filename_score']:.0%} | 页面{r['page_score']:.0%} | 内容{r['content_score']:.0%}
                <br/>答案: {escape_html(r.get('answer', '')[:100])}...
            </li>
        """)

    return f"""
    <ul>
        {"".join(rows)}
    </ul>
    """


def generate_html_report(results: Dict, output_path: str) -> None:
    """
    Generate HTML evaluation report.

    Args:
        results: Evaluation results from RAGEvaluator
        output_path: Output file path
    """
    metrics = results.get("metrics", results)
    detailed = results.get("detailed_results", results.get("detailed_results", []))

    html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>多模态RAG系统评估报告</title>
        <style>
            body {{
                font-family: "Microsoft YaHei", Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{ color: #333; }}
            .score-good {{ color: #28a745; font-weight: bold; }}
            .score-bad {{ color: #dc3545; font-weight: bold; }}
            .score-medium {{ color: #ffc107; }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #333;
            }}
            .metric-label {{
                color: #666;
                font-size: 0.9em;
            }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #007bff; color: white; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            summary {{ cursor: pointer; color: #007bff; }}
            details {{ margin: 5px 0; }}
            .summary-bar {{
                background: #e9ecef;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .summary-item {{
                display: inline-block;
                margin-right: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>多模态RAG系统评估报告</h1>
            <p>生成时间: {get_timestamp()}</p>

            <div class="summary-bar">
                <div class="summary-item">
                    <strong>总分:</strong>
                    <span class="{'score-good' if results['overall_score'] >= 0.75 else 'score-bad'}">
                        {results['overall_score']:.3f}
                    </span>
                </div>
                <div class="summary-item">
                    <strong>完美回答:</strong> {results['perfect_scores']} / {len(detailed)}
                </div>
                <div class="summary-item">
                    <strong>低分回答:</strong> {results['failed_scores']} / {len(detailed)}
                </div>
            </div>

            <h2>指标详情</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value {'score-good' if metrics['avg_filename_score'] >= 0.85 else 'score-bad'}">
                        {metrics['avg_filename_score']:.3f}
                    </div>
                    <div class="metric-label">文件名匹配度 (0.25)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'score-good' if metrics['avg_page_score'] >= 0.80 else 'score-bad'}">
                        {metrics['avg_page_score']:.3f}
                    </div>
                    <div class="metric-label">页面匹配度 (0.25)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'score-good' if metrics['avg_content_score'] >= 0.70 else 'score-bad'}">
                        {metrics['avg_content_score']:.3f}
                    </div>
                    <div class="metric-label">内容相似度 (0.50)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'score-good' if results['overall_score'] >= 0.75 else 'score-bad'}">
                        {results['overall_score']:.3f}
                    </div>
                    <div class="metric-label">综合得分</div>
                </div>
            </div>

            <h2>逐题详情</h2>
            <table>
                <tr>
                    <th>Query ID</th>
                    <th>文件名分</th>
                    <th>页面分</th>
                    <th>内容分</th>
                    <th>总分</th>
                    <th>生成答案</th>
                </tr>
                {generate_table_rows(detailed)}
            </table>

            <h2>失败案例分析</h2>
            {generate_failure_analysis(detailed)}
        </div>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def get_timestamp() -> str:
    """Get current timestamp string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate HTML evaluation report")
    parser.add_argument("results_file", help="Path to evaluation results JSON")
    parser.add_argument("--output", "-o", default="evaluation_report.html", help="Output HTML file")

    args = parser.parse_args()

    with open(args.results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    generate_html_report(results, args.output)
    print(f"Report generated: {args.output}")