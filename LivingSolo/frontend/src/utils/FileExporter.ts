import * as xlsx from "xlsx";
import client from "../store/apis/client";

const downloadBlob = (content : string, filename : string , contentType: string) => {
  // Create a blob
  const blob = new Blob(["\ufeff"+content], { type: contentType });
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
  downloadBlob(response.data, `${name}.csv`, 'text/csv; charset=euc-kr');
};

export const xlsxFileExporter = async (link : string, name: string) => {
  const response = await client.get(link);
  downloadXLSX(response.data, name);
};