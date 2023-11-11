import * as xlsx from "xlsx";
import client from "../store/apis/client";

// Import xlsx -------------------------------------------------------------------------------------------
export type worksheet = xlsx.WorkSheet;
export type workbook = xlsx.WorkBook;
export type xlsxUploadCallback = (workbook : xlsx.WorkBook) => void;
export const xlsxFileImporter = async (e: React.ChangeEvent<HTMLInputElement>, callback: xlsxUploadCallback ) => {
  const reader = new FileReader()
  reader.onload = (e: ProgressEvent<FileReader>) => {
      const data = e.target!.result;
      const workbook = xlsx.read(data, {type: "array"});
      callback(workbook);
  };
  if(e.target && e.target.files && e.target.files.length > 0){
    reader.readAsArrayBuffer(e.target.files[0]);
  }
};
export const xlsxParser = (workSheet: xlsx.WorkSheet) => {
  return xlsx.utils.sheet_to_json(workSheet);
}
// Export xlsx -------------------------------------------------------------------------------------------
const downloadBlob = (content : string, filename : string , contentType: string) => {
  // Create a blob
  const blob = new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);

  // Create a link to download it
  const pom = document.createElement('a');
  pom.href = url;
  pom.setAttribute('download', filename);
  pom.click();
};

const downloadXLSX = (content : string, filename : string) => {
  const stringRows = content.split('\r\n');
  const stringAOA = stringRows.map((row) => row.split(',')); // Array Of Arrays
  const xlsx_sheet = xlsx.utils.aoa_to_sheet(stringAOA);
  const xlsx_book = xlsx.utils.book_new();
  xlsx.utils.book_append_sheet(xlsx_book, xlsx_sheet, 'sheet title!!');
  xlsx.writeFile(xlsx_book, `${filename}.xlsx`);
};

export const csvFileExporter = async (link : string, name: string) => {
  const response = await client.get(link);
  downloadBlob("\ufeff"+response.data, `${name}.csv`, 'text/csv; charset=euc-kr');
};

export const xlsxFileExporter = async (link : string, name: string) => {
  const response = await client.get(link);
  downloadXLSX(response.data, name);
};