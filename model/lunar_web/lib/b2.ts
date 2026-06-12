import { S3Client } from "@aws-sdk/client-s3"

// ⚠️ 자격증명은 환경변수로만. 코드에 키 하드코딩 금지(공개 저장소 유출 방지).
// 참고: 프론트에서 B2를 직접 호출하면 키가 브라우저로 노출됨 → 모든 B2 작업은
// 백엔드(/b2/*) 경유가 원칙. (이 모듈은 현재 미사용)
export const b2Client = new S3Client({
  endpoint: process.env.NEXT_PUBLIC_B2_ENDPOINT ?? "https://s3.us-west-004.backblazeb2.com",
  region: process.env.NEXT_PUBLIC_B2_REGION ?? "us-west-004",
  credentials: {
    accessKeyId: process.env.NEXT_PUBLIC_B2_KEY_ID ?? "",
    secretAccessKey: process.env.NEXT_PUBLIC_B2_APP_KEY ?? "",
  },
})

export const BUCKET_NAME = process.env.NEXT_PUBLIC_B2_BUCKET ?? "lunar-dicom-storage"