import { BootstrapDialog, BootstrapDialogTitle } from "../general/Dialog";
import { styled } from "styled-components";
import { Button, DialogActions, DialogContent } from "@mui/material";
import { workbook, xlsxParser } from "../../utils/FileInterface";
import { useState } from "react";
import { DEFAULT_OPTION } from "../../utils/Constants";

interface TrxnExcelDialogProps {
    open: boolean,
    handleClose: () => void,
    workbook: workbook,
};

export const TrxnExcelDialog = ({open, handleClose, workbook} : TrxnExcelDialogProps) => {
  const workbookSheetNames = workbook.SheetNames;
  const [sheetSelect, setSheetSelect] = useState<string>(DEFAULT_OPTION);
  const SheetContent = () => {
    const parsedSheet = xlsxParser(workbook.Sheets[sheetSelect]);
    const keys_ = Object.keys(parsedSheet[0] as any);
    let [keys, setKeys] = useState<string[]>([...keys_]);
    if(sheetSelect !== DEFAULT_OPTION){
        if(parsedSheet.length > 0){
            console.log(parsedSheet);
            console.log(keys);
            return <>
              <ColNameDiv>
                {keys_.map((key) => <div key={key} onClick={() => {
                  if(keys.some((k) => k === key)){
                    setKeys(keys.filter((k) => key !== k));
                  }else{
                    setKeys(k => [...k, key]);
                  }
                }}>{key} </div>)}
              </ColNameDiv>
                {parsedSheet.map((row: any) => {
                    return <>
                        {keys.map((key) => {
                            return <span key={key}>{row[key] && row[key]}, </span>
                        })}
                        <br />
                    </>
                })}
            </> 
        }
    }
    return <></>
  }
  return <div>
    <BootstrapDialog
      onClose={handleClose}
      aria-labelledby="customized-dialog-title"
      open={open}
      fullWidth={true}
      maxWidth={false}
    >
      <DialogBody>
        <BootstrapDialogTitle id="customized-dialog-title" onClose={handleClose}>
          태그 설정
          <select value={sheetSelect} onChange={(e) => setSheetSelect(e.target.value)}>
              <option disabled value={DEFAULT_OPTION}>
                  - 시트를 선택하세요. -
              </option>
              {workbookSheetNames.map((sheetName) => <option value={sheetName} key={sheetName}>{sheetName}</option>)}
          </select>
        </BootstrapDialogTitle>
        <DialogContent dividers> 
          {sheetSelect !== DEFAULT_OPTION && <SheetContent />}  
        </DialogContent>
        <DialogActions>
          <Button autoFocus onClick={handleClose}>
            닫기
          </Button>
        </DialogActions>
      </DialogBody>
    </BootstrapDialog>
  </div>
};

const DialogBody = styled.div`
  width: 100%;
`;

const ColNameDiv = styled.div`
  display: flex;
  flex-direction: row;
`;