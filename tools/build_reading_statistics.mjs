import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const sourceDir = "C:/Users/Lenovo/Downloads/batch_results";
const sourceCsv = path.join(sourceDir, "reading_statistics.csv");
const conclusionCsv = path.join(sourceDir, "reading_statistics_conclusions.csv");
const outputDir = "D:/WorkStation/project_flapdetection/outputs/reading_statistics_20260914";
const outputXlsx = path.join(outputDir, "reading_statistics_conclusions.xlsx");
const previewPng = path.join(outputDir, "conclusions_preview.png");

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(field); field = ""; }
    else if (ch === '\n') { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += ch;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  return rows.filter(r => r.some(v => v !== ""));
}

function csvEscape(value) {
  const s = value == null ? "" : String(value);
  return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

function num(v) {
  if (v == null || String(v).trim() === "") return null;
  const n = Number(String(v).trim());
  return Number.isFinite(n) ? n : null;
}

function round(v, digits = 3) {
  if (v == null || !Number.isFinite(v)) return "";
  const f = 10 ** digits;
  return Math.round(v * f) / f;
}

function statusFor(row, actual, raw, rectified) {
  if (actual == null) return "missing_actual_reading";
  if (row.pose_status !== "ok") return row.pose_status || "pose_failed";
  if (raw == null && rectified == null) return row.reading_status || "reading_failed";
  if (raw == null) return "unrectified_reading_missing";
  if (rectified == null) return "rectified_reading_missing";
  return "ok";
}

const sourceMatrix = parseCsv((await fs.readFile(sourceCsv, "utf8")).replace(/^\uFEFF/, ""));
const sourceHeaders = sourceMatrix[0].map(x => x.trim());
const sourceRows = sourceMatrix.slice(1).map(cells => Object.fromEntries(sourceHeaders.map((h, i) => [h, cells[i] ?? ""])));

const headers = [
  "image_name", "actual_reading", "unrectified_reading",
  "unrectified_error", "unrectified_absolute_error", "unrectified_accuracy_full_scale_percent",
  "rectified_reading", "rectified_error", "rectified_absolute_error", "rectified_accuracy_full_scale_percent",
  "accuracy_change_percentage_points", "better_method", "result_status", "scale_span"
];

const rows = sourceRows.map(src => {
  const actual = num(src.actual_reading);
  const raw = num(src.unrectified_reading);
  const rectified = num(src.rectified_reading);
  const scaleMin = num(src.scale_min);
  const scaleMax = num(src.scale_max);
  const span = scaleMin != null && scaleMax != null && scaleMax > scaleMin ? scaleMax - scaleMin : null;
  const rawErr = actual != null && raw != null ? raw - actual : null;
  const rectErr = actual != null && rectified != null ? rectified - actual : null;
  const rawAcc = rawErr != null && span ? Math.max(0, 1 - Math.abs(rawErr) / span) : null;
  const rectAcc = rectErr != null && span ? Math.max(0, 1 - Math.abs(rectErr) / span) : null;
  const change = rawAcc != null && rectAcc != null ? rectAcc - rawAcc : null;
  const better = change == null ? "unavailable" : Math.abs(change) < 1e-9 ? "same" : change > 0 ? "rectified" : "unrectified";
  return {
    image_name: src.image_name, actual_reading: actual ?? "", unrectified_reading: raw ?? "",
    unrectified_error: round(rawErr), unrectified_absolute_error: rawErr == null ? "" : round(Math.abs(rawErr)),
    unrectified_accuracy_full_scale_percent: round(rawAcc, 6), rectified_reading: rectified ?? "",
    rectified_error: round(rectErr), rectified_absolute_error: rectErr == null ? "" : round(Math.abs(rectErr)),
    rectified_accuracy_full_scale_percent: round(rectAcc, 6), accuracy_change_percentage_points: round(change, 6),
    better_method: better, result_status: statusFor(src, actual, raw, rectified), scale_span: span ?? "",
  };
});

await fs.mkdir(outputDir, { recursive: true });
const csvLines = [headers.join(","), ...rows.map(r => headers.map(h => csvEscape(r[h])).join(","))];
await fs.writeFile(conclusionCsv, "\uFEFF" + csvLines.join("\r\n"), "utf8");

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const results = workbook.worksheets.add("Results");
const font = "Arial";
const lastRow = rows.length + 1;
summary.showGridLines = false;
results.freezePanes.freezeRows(1);
results.freezePanes.freezeColumns(1);

const matrix = [headers, ...rows.map(r => headers.map(h => {
  if (["image_name", "better_method", "result_status"].includes(h)) return r[h];
  return r[h] === "" ? null : Number(r[h]);
}))];
results.getRangeByIndexes(0, 0, matrix.length, headers.length).values = matrix;
for (let i = 2; i <= lastRow; i++) {
  results.getRange(`D${i}:F${i}`).formulas = [[
    `=IF(COUNT(B${i},C${i})<2,"",C${i}-B${i})`, `=IF(COUNT(D${i})<1,"",ABS(D${i}))`,
    `=IF(COUNT(E${i},N${i})<2,"",MAX(0,1-E${i}/N${i}))`,
  ]];
  results.getRange(`H${i}:L${i}`).formulas = [[
    `=IF(COUNT(B${i},G${i})<2,"",G${i}-B${i})`, `=IF(COUNT(H${i})<1,"",ABS(H${i}))`,
    `=IF(COUNT(I${i},N${i})<2,"",MAX(0,1-I${i}/N${i}))`,
    `=IF(COUNT(F${i},J${i})<2,"",J${i}-F${i})`,
    `=IF(COUNT(K${i})<1,"unavailable",IF(ABS(K${i})<0.0000001,"same",IF(K${i}>0,"rectified","unrectified")))`,
  ]];
}

results.getRange(`A1:N${lastRow}`).format.font = { name: font, size: 10 };
results.getRange("A1:N1").format = { fill: "#1F4E78", font: { name: font, bold: true, color: "#FFFFFF" }, wrapText: true, horizontalAlignment: "center", verticalAlignment: "center" };
results.getRange("A1:N1").format.rowHeight = 42;
results.getRange(`A1:N${lastRow}`).format.borders = { insideHorizontal: { style: "thin", color: "#E2E8F0" }, bottom: { style: "thin", color: "#CBD5E1" } };
results.getRange("A:A").format.columnWidth = 31;
results.getRange("B:K").format.columnWidth = 18;
results.getRange("L:N").format.columnWidth = 22;
results.getRange(`B2:E${lastRow}`).format.numberFormat = "0.000";
results.getRange(`G2:I${lastRow}`).format.numberFormat = "0.000";
results.getRange(`F2:F${lastRow}`).format.numberFormat = "0.0%";
results.getRange(`J2:K${lastRow}`).format.numberFormat = "0.0%";
results.getRange(`F2:F${lastRow}`).conditionalFormats.add("colorScale", { colors: ["#FECACA", "#FEF3C7", "#DCFCE7"], thresholds: ["min", {type:"percentile",value:50}, "max"] });
results.getRange(`J2:J${lastRow}`).conditionalFormats.add("colorScale", { colors: ["#FECACA", "#FEF3C7", "#DCFCE7"], thresholds: ["min", {type:"percentile",value:50}, "max"] });
results.tables.add(`A1:N${lastRow}`, true, "ConclusionResultsTable").style = "TableStyleMedium2";

summary.getRange("A1:H1").values = [["磁翻板液位读数结论", "", "", "", "", "", "", ""]];
summary.getRange("A1:H1").format = { font: { name: font, size: 16, bold: true, color: "#0F172A" }, borders: { bottom: { style: "thin", color: "#94A3B8" } } };
summary.getRange("A2:H2").values = [["准确度 = max(0, 1 - 绝对误差 ÷ 量程跨度)。人工读数缺失或为文字说明的记录不参与准确度统计。", "", "", "", "", "", "", ""]];
summary.getRange("A2:H2").format = { font: { name: font, size: 10, italic: true, color: "#475569" } };
summary.getRange("A4:C11").values = [
  ["统计项", "未透视", "透视后"], ["可评估样本", null, null], ["平均绝对误差 MAE", null, null],
  ["均方根误差 RMSE", null, null], ["平均准确度（%FS）", null, null], ["绝对误差 ≤ 1", null, null],
  ["绝对误差 ≤ 2", null, null], ["绝对误差 ≤ 5", null, null],
];
summary.getRange("B5:C11").formulas = [
  [`=COUNT(Results!E2:E${lastRow})`, `=COUNT(Results!I2:I${lastRow})`],
  [`=AVERAGE(Results!E2:E${lastRow})`, `=AVERAGE(Results!I2:I${lastRow})`],
  [`=SQRT(SUMSQ(Results!D2:D${lastRow})/COUNT(Results!D2:D${lastRow}))`, `=SQRT(SUMSQ(Results!H2:H${lastRow})/COUNT(Results!H2:H${lastRow}))`],
  [`=AVERAGE(Results!F2:F${lastRow})`, `=AVERAGE(Results!J2:J${lastRow})`],
  [`=(COUNTIF(Results!E2:E${lastRow},"<=1")-COUNTBLANK(Results!E2:E${lastRow}))/COUNT(Results!E2:E${lastRow})`, `=(COUNTIF(Results!I2:I${lastRow},"<=1")-COUNTBLANK(Results!I2:I${lastRow}))/COUNT(Results!I2:I${lastRow})`],
  [`=(COUNTIF(Results!E2:E${lastRow},"<=2")-COUNTBLANK(Results!E2:E${lastRow}))/COUNT(Results!E2:E${lastRow})`, `=(COUNTIF(Results!I2:I${lastRow},"<=2")-COUNTBLANK(Results!I2:I${lastRow}))/COUNT(Results!I2:I${lastRow})`],
  [`=(COUNTIF(Results!E2:E${lastRow},"<=5")-COUNTBLANK(Results!E2:E${lastRow}))/COUNT(Results!E2:E${lastRow})`, `=(COUNTIF(Results!I2:I${lastRow},"<=5")-COUNTBLANK(Results!I2:I${lastRow}))/COUNT(Results!I2:I${lastRow})`],
];
summary.getRange("E4:F10").values = [["结论项", "数值"], ["总图片数", null], ["有效人工读数", null], ["两种方法均可比较", null], ["透视后更准确", null], ["未透视更准确", null], ["两者相同", null]];
summary.getRange("F5:F10").formulas = [[`=COUNTA(Results!A2:A${lastRow})`], [`=COUNT(Results!B2:B${lastRow})`], [`=COUNT(Results!K2:K${lastRow})`], [`=COUNTIF(Results!L2:L${lastRow},"rectified")`], [`=COUNTIF(Results!L2:L${lastRow},"unrectified")`], [`=COUNTIF(Results!L2:L${lastRow},"same")`]];
for (const range of ["A4:C11", "E4:F10"]) {
  summary.getRange(range).format.font = { name: font, size: 10 };
  summary.getRange(range).format.borders = { insideHorizontal: { style: "thin", color: "#E2E8F0" }, bottom: { style: "thin", color: "#CBD5E1" } };
}
summary.getRange("A4:C4").format = { fill: "#1F4E78", font: { name: font, bold: true, color: "#FFFFFF" } };
summary.getRange("E4:F4").format = { fill: "#1F4E78", font: { name: font, bold: true, color: "#FFFFFF" } };
summary.getRange("B6:C7").format.numberFormat = "0.000";
summary.getRange("B8:C11").format.numberFormat = "0.0%";
summary.getRange("A:A").format.columnWidth = 27;
summary.getRange("B:C").format.columnWidth = 18;
summary.getRange("E:E").format.columnWidth = 25;
summary.getRange("F:F").format.columnWidth = 15;
const preview = await workbook.render({ sheetName: "Summary", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(previewPng, new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputXlsx);
const inspection = await workbook.inspect({ kind: "table", sheetId: "Summary", range: "A1:O18", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 15, maxChars: 10000 });
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
await Promise.all([
  fs.rm(path.join(sourceDir, "reading_statistics_enriched.csv"), { force: true }), fs.rm(path.join(outputDir, "reading_statistics_analysis.xlsx"), { force: true }),
  fs.rm(path.join(outputDir, "boundary_y_illustration.png"), { force: true }), fs.rm(path.join(outputDir, "boundary_sheet_preview.png"), { force: true }), fs.rm(path.join(outputDir, "summary_preview.png"), { force: true }),
]);
console.log(JSON.stringify({ outputXlsx, conclusionCsv, previewPng, inspection: inspection.ndjson, errors: errors.ndjson }, null, 2));
